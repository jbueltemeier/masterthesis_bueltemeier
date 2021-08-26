from abc import abstractmethod
from typing import Dict,  Sequence, cast

import torch

import pystiche
from pystiche import enc


__all__ = [
    "_Transformer",
]





class _Transformer(pystiche.Module):
    r"""Abstract base class for all Transformers.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        decoder: Decoder that is used to decode the encodings to an output image.
    """

    def __init__(self, encoder: enc.Encoder, decoder: SequentialDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if encoder.layer != decoder.layer:
            msg = (
                f"The layer of the encoder {encoder.layer} and decoder {decoder.layer}"
                " do not match."
            )
            raise AttributeError(msg)

    @abstractmethod
    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def input_image_to_enc(self, image: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(image))

    def enc_to_output_image(self, enc: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(enc))

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.process_input_image(input_image)


class _ConvertTransformer(_Transformer):
    r"""Abstract base class for all Transformers converting in an encoded space."""

    def set_target_image(self, image: torch.Tensor, region: str = "") -> None:
        with torch.no_grad():
            enc = self.target_image_to_enc(image)
            self.target_enc_to_repr(enc, region=region)

        self.register_buffer(f"{region}_target_image", image, persistent=False)

    def target_image_to_enc(self, image: torch.Tensor) -> torch.Tensor:
        return self.input_image_to_enc(image)

    def has_target_image(self, region: str = "") -> bool:
        return f"{region}_target_image" in self._buffers

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_enc = self.input_image_to_enc(image)
        converted_enc = self.convert(input_enc)
        return self.enc_to_output_image(converted_enc)

    @abstractmethod
    def input_enc_to_repr(self, image: torch.Tensor, region: str = "") -> torch.Tensor:
        pass

    @abstractmethod
    def target_enc_to_repr(self, image: torch.Tensor, region: str = "") -> None:
        pass

    @abstractmethod
    def convert(self, enc: torch.Tensor) -> torch.Tensor:
        pass


class _RegionConvertTransformer(_ConvertTransformer):
    r"""Abstract base class for all ConvertTransformers converting with regions."""
    target_enc_guide: torch.Tensor
    input_enc_guide: torch.Tensor

    def __init__(
            self, encoder: enc.Encoder, decoder: SequentialDecoder, regions: Sequence[str],
    ) -> None:
        super().__init__(encoder, decoder)
        self.regions = regions

    def verify_regions(self, given_regions: Sequence[str]) -> None:
        if not all(key in given_regions for key in self.regions):
            msg = (
                f"This autoencoder requires {self.regions} but only {given_regions} "
                f"are given. The missing regions are: "
                f"{set(self.regions).difference(given_regions)}"
            )
            raise TypeError(msg)

    def set_target_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        self.verify_regions(list(guides.keys()))
        for region in self.regions:
            self.set_target_guide(guides[region], region, recalc_enc=True)

    def set_target_guide(
            self, guide: torch.Tensor, region: str, recalc_enc: bool = True
    ) -> None:

        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer(f"{region}_target_guide", guide, persistent=False)
        self.register_buffer(f"{region}_target_enc_guide", enc_guide, persistent=False)
        if recalc_enc and self.has_target_image(region):
            self.set_target_image(getattr(self, f"{region}_target_image"))

    def set_target_images(self, images: Dict[str, torch.Tensor]) -> None:
        self.verify_regions(list(images.keys()))
        for region in self.regions:
            self.set_target_image(images[region], region)

    def set_input_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        for region, guide in guides.items():
            self.set_input_guide(guide, region)

    def set_input_guide(self, guide: torch.Tensor, region: str) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer(f"{region}_input_guide", guide, persistent=False)
        self.register_buffer(f"{region}_input_enc_guide", enc_guide, persistent=False)

    def has_target_guide(self, region: str) -> bool:
        return f"{region}_target_guide" in self._buffers

    def has_input_guide(self, region: str) -> bool:
        return f"{region}_input_guide" in self._buffers