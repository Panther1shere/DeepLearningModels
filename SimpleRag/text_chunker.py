class TextChunker:
    def __init__(self, chunk_size=300, overlap=50):
        """
        :param chunk_size: number of characters in each chunk
        :param overlap: number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """
        Splits the input text into chunks with optional overlap.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap

        return chunks