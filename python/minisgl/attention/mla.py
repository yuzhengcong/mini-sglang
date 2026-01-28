from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .fa import FlashAttentionBackend, FAMetadata, _fa_sgl_impl

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


class MLAFlashAttentionBackend(FlashAttentionBackend):

    def __init__(self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
        super().__init__(config, kvcache, page_table)
        self.latent_dim = getattr(config, "latent_dim", None)
        self.head_dim = config.head_dim
        self.kv_down: torch.Tensor | None = None
        self.k_up: torch.Tensor | None = None
        self.v_up: torch.Tensor | None = None

    def _maybe_compress(self, x: torch.Tensor, is_value: bool = False) -> torch.Tensor:
        W = self.kv_down
        latent_dim = self.latent_dim
        if latent_dim is None or latent_dim == self.head_dim:
            return x
        if W is None:
            raise RuntimeError("kv_down is required when latent_dim != head_dim")
        if W.ndim == 2:
            return torch.matmul(x, W)
        else:
            return torch.einsum("bhd,hdl->bhl", x, W)

    def _maybe_decompress(self, x: torch.Tensor, is_value: bool = False) -> torch.Tensor:
        latent_dim = self.latent_dim
        if latent_dim is None or latent_dim == self.head_dim:
            return x
        W = self.v_up if is_value else self.k_up
        if W is None:
            raise RuntimeError("k_up/v_up is required when latent_dim != head_dim")
        if W.ndim == 2:
            return torch.matmul(x, W)
        else:
            return torch.einsum("pihl,hld->pihd", x, W)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)

        k_lat = self._maybe_compress(k, is_value=False)
        v_lat = self._maybe_compress(v, is_value=True)
        self.kvcache.store_kv(k_lat, v_lat, batch.out_loc, layer_id)

        k_lat_cache = self.kvcache.k_cache(layer_id)
        v_lat_cache = self.kvcache.v_cache(layer_id)
        k_full = self._maybe_decompress(k_lat_cache, is_value=False)
        v_full = self._maybe_decompress(v_lat_cache, is_value=True)

        return _fa_sgl_impl(
            q=q,
            k_cache=k_full,
            v_cache=v_full,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
        )
