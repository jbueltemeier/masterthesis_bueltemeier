from abc import abstractmethod
from typing import Dict,  Sequence, cast

import torch

import pystiche
from pystiche import enc
from pystiche_papers.bueltemeier_mst_2021._modules import Inspiration, SequentialDecoder, encoder, decoder, bottleneck
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale
from pystiche_papers.bueltemeier_mst_2021._modules_johnson_structure import encoder as johnson_encoder
from pystiche_papers.bueltemeier_mst_2021._modules_johnson_structure import decoder as johnson_decoder
from pystiche_papers.bueltemeier_mst_2021._modules_johnson_structure import bottleneck as johnson_bottleneck
__all__ = [
    "_Transformer",
    "_ConvertTransformer",
    "_RegionConvertTransformer",
    "MSTTransformer",
    "MaskMSTTransformer"
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

    @abstractmethod
    def process_input_image(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def input_image_to_enc(self, image: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(image))

    def enc_to_output_image(self, enc: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(enc))

    def forward(self, input_image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.process_input_image(input_image, *args, **kwargs)


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

    @abstractmethod
    def process_input_image(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def input_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        pass

    @abstractmethod
    def target_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> None:
        pass

    @abstractmethod
    def convert(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        pass


class ConvertTransformer(_ConvertTransformer):
    def process_input_image(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        input_repr = self.input_enc_to_repr(self.input_image_to_enc(image))
        converted_enc = self.convert(input_repr)
        output_image = self.enc_to_output_image(converted_enc)
        if pystiche.image.extract_num_channels(output_image) == 1:
            output_image = grayscale_to_fakegrayscale(output_image)
        return output_image

    @abstractmethod
    def input_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        pass

    @abstractmethod
    def target_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> None:
        pass

    @abstractmethod
    def convert(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
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

    def set_target_guide(
            self, guide: torch.Tensor, region: str, recalc_enc: bool = True
    ) -> None:

        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer(f"{region}_target_guide", guide)
        self.register_buffer(f"{region}_target_enc_guide", enc_guide)
        if recalc_enc and self.has_target_image(region):
            self.set_target_image(getattr(self, f"{region}_target_image"), region=region)

    def set_input_guides(self, guides: Dict[str, torch.Tensor]) -> None:
        for region, guide in guides.items():
            self.set_input_guide(guide, region)

    def set_input_guide(self, guide: torch.Tensor, region: str) -> None:
        with torch.no_grad():
            enc_guide = self.encoder.propagate_guide(guide)
        self.register_buffer(f"{region}_input_guide", guide)
        self.register_buffer(f"{region}_input_enc_guide", enc_guide)

    def has_target_guide(self, region: str) -> bool:
        return f"{region}_target_guide" in self._buffers

    def has_input_guide(self, region: str) -> bool:
        return f"{region}_input_guide" in self._buffers

    @staticmethod
    def apply_guide(image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        r"""Apply a guide to an image.

        Args:
            image: Image of shape :math:`B \times C \times H \times W`.
            guide: Guide of shape :math:`1 \times 1 \times H \times W`.
        """
        return image * guide

    @abstractmethod
    def convert(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        pass


class RegionConvertTransformer(_RegionConvertTransformer):
    def process_input_image(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        regions = args[0]
        input_enc = self.input_image_to_enc(image)
        converted_enc = []
        for region in regions:
            input_repr = self.input_enc_to_repr(input_enc, region=region)
            transformed_enc = self.convert(input_repr, region=region)
            if self.has_input_guide(region):
                transformed_enc = self.apply_guide(transformed_enc, getattr(self, f"{region}_input_enc_guide"))
            converted_enc.append(transformed_enc)

        converted_enc = torch.sum(torch.stack(converted_enc), dim=0)
        # converted_enc = self._bottleneck(converted_enc)
        return self.enc_to_output_image(converted_enc)

    @abstractmethod
    def input_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        pass

    @abstractmethod
    def target_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> None:
        pass

    @abstractmethod
    def convert(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        pass


class MSTTransformer(ConvertTransformer):
    def __init__(self, in_channels=3, instance_norm=False) -> None:
        channels = 32
        expansion = 4
        _encoder = encoder(in_channels=in_channels, channels=channels, expansion=expansion, instance_norm=instance_norm)
        _decoder = decoder(channels, out_channels=in_channels, expansion=expansion, instance_norm=instance_norm)
        # _encoder = johnson_encoder(in_channels)
        # _decoder = johnson_decoder(channels, in_channels)
        super().__init__(_encoder, _decoder)
        self.inspiration = Inspiration(channels * expansion)
        self._bottleneck = bottleneck(channels, expansion=expansion, instance_norm=instance_norm)
        # self._bottleneck = johnson_bottleneck(channels)

    def input_enc_to_repr(self, enc: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return enc

    def target_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> None:
        target_repr = pystiche.gram_matrix(enc, normalize=True)
        self.register_buffer(f"_target_repr", target_repr)
        self.inspiration.setTarget(target_repr)

    def convert(self, enc: torch.Tensor, region: str = "", recalc_enc: bool = True) -> torch.Tensor:
        # Update target enc during training due to changing encoder
        if recalc_enc and self.has_target_image(region):
            self.set_target_image(getattr(self, f"{region}_target_image"), region=region)
        converted_enc = self.inspiration(enc)
        return self._bottleneck(converted_enc)


class MaskMSTTransformer(RegionConvertTransformer):
    def __init__(self, regions: Sequence[str], in_channels=3, instance_norm=False) -> None:
        channels = 32
        expansion = 4
        _encoder = encoder(in_channels=in_channels, channels=channels, expansion=expansion, instance_norm=instance_norm)
        _decoder = decoder(channels, out_channels=in_channels, expansion=expansion, instance_norm=instance_norm)
        super().__init__(_encoder, _decoder, regions=regions)
        for region in regions:
            setattr(self, f"{region}_inspiration", Inspiration(channels * expansion))
            setattr(self, f"{region}_bottleneck", bottleneck(channels, expansion=expansion, instance_norm=instance_norm))
        # self._bottleneck = bottleneck(channels, expansion=expansion, instance_norm=instance_norm)

    def input_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> torch.Tensor:
        inpur_repr = enc
        if self.has_input_guide(region):
            inpur_repr = self.apply_guide(enc, getattr(self, f"{region}_input_enc_guide"))
        return inpur_repr

    def target_enc_to_repr(self, enc: torch.Tensor, region: str = "") -> None:
        if self.has_target_guide(region):
            guide = getattr(self, f"{region}_target_enc_guide")
            enc = self.apply_guide(enc, guide)
            target_repr = pystiche.gram_matrix(enc, normalize=False) / torch.sum(guide)
            self.register_buffer(f"{region}_target_repr", target_repr)
        else:
            target_repr = pystiche.gram_matrix(enc, normalize=True)
            self.register_buffer(f"{region}_target_repr", target_repr)

    def convert(self, enc: torch.Tensor, region: str = "", recalc_enc: bool = True) -> torch.Tensor:
        # Update target enc during training due to changing encoder
        if recalc_enc and self.has_target_image(region):
            self.set_target_image(getattr(self, f"{region}_target_image"), region=region)
        getattr(self, f"{region}_inspiration").setTarget(getattr(self, f"{region}_target_repr"))
        transformed_enc = getattr(self, f"{region}_inspiration")(enc)
        return getattr(self, f"{region}_bottleneck")(transformed_enc)
