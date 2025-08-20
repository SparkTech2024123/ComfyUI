import torch


class ImageBatchExtender:
    """
    Extends an image batch by duplicating the first and last images.

    This node takes an image batch and adds copies of the first image to the beginning
    and copies of the last image to the end of the batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prepend_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of copies of the first image to add at the beginning"
                }),
                "append_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of copies of the last image to add at the end"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("extended_images",)
    FUNCTION = "extend_batch"
    CATEGORY = "image/batch"

    def extend_batch(self, images, prepend_count, append_count):
        """
        Extend the image batch by duplicating first and last images.

        Args:
            images: Input image batch tensor with shape [B, H, W, C]
            prepend_count: Number of first image copies to add at beginning
            append_count: Number of last image copies to add at end

        Returns:
            Extended image batch tensor
        """
        # Validate input
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, H, W, C], got {images.dim()}D tensor")

        batch_size = images.shape[0]
        if batch_size == 0:
            raise ValueError("Input batch is empty")

        # Start with the original images
        result_parts = [images]

        # Prepend copies of the first image
        if prepend_count > 0:
            first_image = images[0:1]  # Keep batch dimension [1, H, W, C]
            prepend_images = first_image.repeat(prepend_count, 1, 1, 1)
            result_parts.insert(0, prepend_images)

        # Append copies of the last image
        if append_count > 0:
            last_image = images[-1:]  # Keep batch dimension [1, H, W, C]
            append_images = last_image.repeat(append_count, 1, 1, 1)
            result_parts.append(append_images)

        # Concatenate all parts along the batch dimension
        extended_batch = torch.cat(result_parts, dim=0)

        return (extended_batch,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ImageBatchExtender": ImageBatchExtender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchExtender": "Image Batch Extender",
}