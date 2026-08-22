from abc import ABC, abstractmethod


class VisionService(ABC):
    @abstractmethod
    def analyze_image(self, image_bytes: bytes, question: str) -> str: ...

