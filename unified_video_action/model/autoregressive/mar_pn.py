from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
from einops import rearrange
import torch

from unified_video_action.model.autoregressive.mar_con_unified import MAR


def mask_by_order(mask_len, order, bsz, seq_len, device):
    masking = torch.zeros(bsz, seq_len).to(device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).to(device),
    ).bool()
    return masking


class MAR_PN(MAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_tokens(
        self,
        bsz,
        cond,
        text_latents=None,
        num_iter=64,
        cfg=1.0,
        cfg_negative=1.0,
        cfg_schedule="linear",
        temperature=1.0,
        progress=False,
        history_nactions=None,
        nactions=None,
        proprioception_input={},
        task_mode=None,
        vae_model=None,
        x=None,
    ):
        self.device = cond.device
        B, T, C, H, W = cond.size()
        cond = rearrange(cond, "b t c h w -> (b t) c h w")
        cond = self.patchify(cond)
        cond = rearrange(
            cond, "(b t) seq_len c -> b t seq_len c", b=B
        )

        # ========= Proprioception =========
        if self.use_proprioception:
            if "second_image_z" in proprioception_input:
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"], "b t c h w -> (b t) c h w"
                )
                proprioception_input["second_image_z"] = self.patchify(
                    proprioception_input["second_image_z"]
                )
                proprioception_input["second_image_z"] = rearrange(
                    proprioception_input["second_image_z"],
                    "(b t) seq_len c -> b t seq_len c",
                    b=B,
                )

        if text_latents is not None and hasattr(self, "text_proj_cond"):
            if self.language_emb_model_type == 1:
                text_latents = self.text_proj_cond(text_latents)

        # ========= Mask =========
        if task_mode == "inverse_model":
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.patchify(x)
            tokens = rearrange(
                x, "(b t) seq_len c -> b t seq_len c", b=B
            )
            mask = torch.zeros(bsz, self.n_frames, self.seq_len).to(self.device)
        else:
            # init and sample generation orders
            tokens = torch.zeros(
                bsz, self.n_frames, self.seq_len, self.token_embed_dim
            ).to(self.device)
            mask = torch.ones(bsz, self.n_frames, self.seq_len).to(self.device)
            if self.predict_wrist_img:
                proprioception_input["pred_second_image_z"] = torch.zeros(
                    bsz, self.n_frames, self.seq_len, self.token_embed_dim
                ).to(self.device)

        # ========= Sample Orders =========
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # ========= Predict Video =========
        if self.predict_video:
            for step in indices:
                cur_tokens = tokens.clone()

                if self.predict_wrist_img:
                    cur_wrist_tokens = proprioception_input[
                        "pred_second_image_z"
                    ].clone()
                
                tokens = torch.cat([tokens, tokens, tokens], dim=0)
                mask = torch.cat([mask, mask, mask], dim=0)

                x = self.forward_mae_encoder(
                    tokens,
                    mask,
                    cond,
                    text_latents,
                    history_nactions=history_nactions,
                    nactions=nactions,
                    task_mode=task_mode,
                    proprioception_input=proprioception_input,
                )
                z = self.forward_mae_decoder(x, mask)

                if self.predict_action:
                    sampled_token_latent_act = self.diffactloss.sample(
                        z, temperature, cfg=cfg, text_latents=text_latents, pos_neg_sample=True, cfg_negative=cfg_negative
                    )
                    sampled_token_latent_act, _, _ = sampled_token_latent_act.chunk(3, dim=0)  # Remove null class samples
                else:
                    sampled_token_latent_act = None

                # ========= Predict action and return if task_mode is inverse_model or policy_model=========
                if task_mode == "inverse_model" or task_mode == "policy_model":
                    return None, sampled_token_latent_act

                # ========= Mask Ratio =========
                # mask ratio for the next round, following MaskGIT and MAGE.
                mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).to(
                    self.device
                )

                # take the first frame mask
                mask_ = mask[:, 0]

                # masks out at least one for the next iteration
                mask_len = torch.maximum(
                    torch.Tensor([1]).to(self.device),
                    torch.minimum(
                        torch.sum(mask_, dim=-1, keepdims=True) - 1, mask_len
                    ),
                )

                # get masking for next iteration and locations to be predicted in this iteration
                mask_next = mask_by_order(
                    mask_len[0], orders, bsz, self.seq_len, self.device
                )

                ## expand mask_next to all frames
                mask_next = mask_next.unsqueeze(1).expand(-1, T, -1)
                mask_next = rearrange(mask_next, "b t s -> b (t s)")
                mask = rearrange(mask, "b t s -> b (t s)")

                if step >= num_iter - 1:
                    mask_to_pred = mask[:bsz].bool()
                else:
                    mask_to_pred = torch.logical_xor(
                        mask[:bsz].bool(), mask_next.bool()
                    )
                mask = mask_next
                mask = rearrange(mask, "b (t s) -> b t s", t=self.n_frames)

                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred, mask_to_pred], dim=0)

                # sample token latents for this step
                z = z[mask_to_pred.nonzero(as_tuple=True)]
                # cfg schedule follow Muse
                if cfg_schedule == "linear":
                    cfg_iter = (
                        1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                    )
                    cfg_negative_iter = (
                        1 + (cfg_negative - 1) * (self.seq_len - mask_len[0]) / self.seq_len
                    )
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                    cfg_negative_iter = cfg_negative
                else:
                    raise NotImplementedError

                sampled_token_latent = self.diffloss.sample(
                    z, temperature, cfg_iter, text_latents=text_latents, pos_neg_sample=True, cfg_negative=cfg_negative_iter
                )

                sampled_token_latent, _, _ = sampled_token_latent.chunk(
                    3, dim=0
                )  # Remove null class samples
                mask_to_pred, _, _ = mask_to_pred.chunk(3, dim=0)

                cur_tokens = rearrange(cur_tokens, "b t s c -> b (t s) c")
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
                cur_tokens = rearrange(
                    cur_tokens, "b (t s) c -> b t s c", t=self.n_frames
                )
                tokens = cur_tokens.clone()

                # ========= Predict Wrist Image =========
                if self.predict_wrist_img:
                    sampled_wrist_token_latent = self.diffloss_wrist.sample(
                        z, temperature, cfg_iter, text_latents=text_latents
                    )

                    sampled_wrist_token_latent, _ = (
                        sampled_wrist_token_latent.chunk(3, dim=0)
                    )  # Remove null class samples

                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b t s c -> b (t s) c"
                    )
                    cur_wrist_tokens[mask_to_pred.nonzero(as_tuple=True)] = (
                        sampled_wrist_token_latent
                    )
                    cur_wrist_tokens = rearrange(
                        cur_wrist_tokens, "b (t s) c -> b t s c", t=self.n_frames
                    )
                    proprioception_input["pred_second_image_z"] = (
                        cur_wrist_tokens.clone()
                    )

            # ========= Unpatchify =========
            tokens = rearrange(tokens, "b t s c -> (b t) s c")
            tokens = self.unpatchify(tokens)
            # tokens = rearrange(tokens, '(b t) c h w -> b t c h w', b=B)

            if self.predict_wrist_img:
                wrist_tokens = rearrange(
                    proprioception_input["pred_second_image_z"], "b t s c -> (b t) s c"
                )
                wrist_tokens = self.unpatchify(wrist_tokens)

        else:
            raise NotImplementedError

        if self.predict_wrist_img:
            return wrist_tokens, sampled_token_latent_act
        else:
            return tokens, sampled_token_latent_act
