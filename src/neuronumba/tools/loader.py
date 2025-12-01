import os

import numpy as np

import neuronumba.tools.hdf as hdf

import numpy as np
import re

def read_csv_with_repeated_delimiters(file_path, delimiter=',', encoding='utf-8', dtype=None, skip_header=False):
    """
    Read a CSV file with repeated delimiters using pure Python/NumPy (no pandas dependency).

    This function handles CSV files where delimiters appear consecutively multiple times
    by parsing the text directly and cleaning the data manually using only built-in Python
    functions and NumPy.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    delimiter : str, default ','
        The delimiter character that may be repeated (e.g., ',', ';', '|', etc.)
    encoding : str, default 'utf-8'
        File encoding
    dtype : data type, optional
        Data type for the resulting array. If None, will automatically infer types.
        Can be: int, float, str, or any numpy dtype
    skip_header : bool, default False
        Whether to skip the first row (assumed to be header)

    Returns:
    --------
    numpy.ndarray
        The data as a numpy array with appropriate data types

    Examples:
    ---------
    >>> # File with repeated commas: "1,,2,,,3\n4,,,5,,6"
    >>> data = read_csv_with_repeated_delimiters_pure('data.csv')
    >>> print(data)
    [[1 2 3]
     [4 5 6]]

    >>> # Mixed data types
    >>> data = read_csv_with_repeated_delimiters_pure('mixed.csv', skip_header=True)
    >>> print(data)  # Returns object array for mixed types

    >>> # Force specific type
    >>> data = read_csv_with_repeated_delimiters_pure('numbers.csv', dtype=float)
    """

    def clean_and_split_line(line, delimiter):
        """Clean a line by removing repeated delimiters and split into fields"""
        if not line.strip():
            return []

        # Remove leading/trailing whitespace
        line = line.strip()

        # Replace multiple consecutive delimiters with single delimiter
        # Need to escape special regex characters
        escaped_delimiter = re.escape(delimiter)
        cleaned_line = re.sub(f"{escaped_delimiter}+", delimiter, line)

        # Remove leading/trailing delimiters if they exist
        cleaned_line = cleaned_line.strip(delimiter)

        # Split by delimiter and clean each field
        if cleaned_line:
            fields = cleaned_line.split(delimiter)
            # Strip whitespace from each field
            fields = [field.strip() for field in fields]
            return fields
        else:
            return []

    def infer_field_type(value):
        """
        Try to infer the most appropriate type for a field value.
        Returns (converted_value, type_class)
        """
        value = value.strip()

        # Handle empty values
        if not value:
            return value, str

        # Try integer first (most restrictive)
        try:
            # Handle values like "3.0" that should be integers
            if '.' in value:
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val), int
                else:
                    return float_val, float
            else:
                return int(value), int
        except ValueError:
            pass

        # Try float
        try:
            return float(value), float
        except ValueError:
            pass

        # Keep as string if nothing else works
        return value, str

    def determine_column_types(data_rows):
        """
        Analyze all data to determine the best type for each column.
        Uses hierarchy: int -> float -> str (can only upgrade, not downgrade)
        """
        if not data_rows or not data_rows[0]:
            return []

        num_cols = len(data_rows[0])
        column_types = [int] * num_cols  # Start with most restrictive type

        # Examine each value to determine column types
        for row in data_rows:
            for col_idx in range(min(len(row), num_cols)):
                value = row[col_idx]
                _, field_type = infer_field_type(value)

                # Upgrade type if necessary (int -> float -> str)
                current_type = column_types[col_idx]
                if current_type == int and field_type == float:
                    column_types[col_idx] = float
                elif current_type in [int, float] and field_type == str:
                    column_types[col_idx] = str

        return column_types

    def convert_value(value, target_type):
        """Convert a single value to the specified target type"""
        value = value.strip()

        # Handle empty values
        if not value:
            if target_type == int:
                return 0
            elif target_type == float:
                return 0.0
            else:
                return ''

        try:
            if target_type == int:
                # Handle "3.0" -> 3 conversion
                return int(float(value))
            elif target_type == float:
                return float(value)
            else:
                return str(value)
        except (ValueError, TypeError):
            # Fallback to string representation if conversion fails
            return str(value)

    def pad_rows(data_rows):
        """Ensure all rows have the same number of columns by padding with empty strings"""
        if not data_rows:
            return data_rows

        max_cols = max(len(row) for row in data_rows)

        for row in data_rows:
            while len(row) < max_cols:
                row.append('')

        return data_rows, max_cols

    # Main processing logic
    try:
        # Read the entire file
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.readlines()

        if not lines:
            raise ValueError("File is empty")

        # Process lines into structured data
        data_rows = []
        start_line = 1 if skip_header else 0

        for line_num, line in enumerate(lines):
            if line_num < start_line:
                continue  # Skip header if requested

            fields = clean_and_split_line(line, delimiter)
            if fields:  # Only add non-empty rows
                data_rows.append(fields)

        if not data_rows:
            raise ValueError("No data rows found after processing")

        # Ensure rectangular data (all rows same length)
        data_rows, num_cols = pad_rows(data_rows)

        # Determine data types for each column
        if dtype is None:
            # Auto-infer types
            column_types = determine_column_types(data_rows)
        else:
            # Use specified type for all columns
            column_types = [dtype] * num_cols

        # Convert all values to their target types
        processed_data = []
        for row in data_rows:
            processed_row = []
            for col_idx, value in enumerate(row):
                if col_idx < len(column_types):
                    converted_value = convert_value(value, column_types[col_idx])
                else:
                    # Fallback for extra columns
                    converted_value = str(value)
                processed_row.append(converted_value)
            processed_data.append(processed_row)

        # Create numpy array with appropriate dtype
        if dtype is not None:
            # Use explicitly specified dtype
            result = np.array(processed_data, dtype=dtype)
        else:
            # Use inferred types
            if len(set(column_types)) == 1:
                # All columns have the same type
                result = np.array(processed_data, dtype=column_types[0])
            else:
                # Mixed types - use object array
                result = np.array(processed_data, dtype=object)

        return result

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error processing file {file_path}: {str(e)}")




def load_2d_matrix(filename, delimiter=None, index=None):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.csv':
        return read_csv_with_repeated_delimiters(filename, delimiter=delimiter)
    elif file_extension == '.tsv':
        if delimiter is None:
            delimiter = '\t'
        return read_csv_with_repeated_delimiters(filename, delimiter=delimiter)
    elif file_extension == '.mat':
        if index is None:
            raise RuntimeError("You have to provide an index for the file")
        return hdf.loadmat(filename)[index]
    elif file_extension == '.npy':
        return np.load(filename)
    elif file_extension == '.npz':
        return np.load(filename, allow_pickle=True)[index]
    else:
        raise RuntimeError("Unrecognized file extension")