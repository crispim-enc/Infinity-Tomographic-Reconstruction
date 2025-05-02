#  Copyright (c) 2025. # *******************************************************
#  * FILE: $FILENAME
#  * AUTHOR: Pedro Encarnação
#  * DATE: $CURRENT_DATE
#  * LICENSE: Your License Name
#  *******************************************************



class GATEMacParser:
    """
    A class to parse and manipulate GATE macro files.
    """

    def __init__(self, file_path):
        """
        Initialize the parser with the GATE macro file.
        Args:
            file_path (str): Path to the GATE macro file.
        """
        self.file_path = file_path
        self.content = self._load_file()

    def _load_file(self):
        """
        Load the content of the macro file.
        Returns:
            list: Lines of the file as a list.
        """
        try:
            with open(self.file_path, 'r') as file:
                return file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

    def get_fields(self, keyword):
        """
        Retrieve all lines containing a specific keyword.
        Args:
            keyword (str): The keyword to search for in the file.
        Returns:
            list: A list of lines containing the keyword.
        """
        return [line.strip() for line in self.content if keyword in line]

    def replace_field(self, keyword, new_value):
        """
        Replace the value of a specific field in the macro file.
        Args:
            keyword (str): The keyword of the field to replace.
            new_value (str): The new value to set.
        """
        updated = False
        for i, line in enumerate(self.content):
            if keyword in line:
                self.content[i] = f"{keyword} {new_value}\n"
                updated = True
        if not updated:
            raise ValueError(f"Keyword '{keyword}' not found in the file.")

    def save(self, output_path):
        """
        Save the modified content to a new file.
        Args:
            output_path (str): Path to save the modified file.
        """
        with open(output_path, 'w') as file:
            file.writelines(self.content)
        print(f"File saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize the parser
    parser = GATEMacParser("path_to_your_gate_file.mac")

    # Retrieve all lines with a specific keyword
    fields = parser.get_fields("/gate/geometry/setMaterialDatabase")
    print("Fields with the keyword:", fields)

    # Replace a specific field value
    parser.replace_field("/gate/world/geometry/setXLength", "30. cm")

    # Save the modified file
    parser.save("modified_gate_file.mac")