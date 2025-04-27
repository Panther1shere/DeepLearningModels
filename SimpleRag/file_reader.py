class FileReader:
    def load_text_file(self, file_path):

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content # Returns: str: Content of the file as a single string.

