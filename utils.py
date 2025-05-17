"""
Utility functions and classes for the evaluation scripts.
"""

class OutputCapture:
    """
    Utility class for capturing output to both console and file.
    
    This class provides a way to print output to both the console and a file
    simultaneously, making it easy to save evaluation results while still
    displaying them to the user.
    """
    def __init__(self, output_file=None):
        """
        Initialize the OutputCapture with an optional output file.
        
        Args:
            output_file (str, optional): Path to the file where output should be saved.
                                         If None, output is only printed to console.
        """
        self.output_file = output_file
        self.file = None
        
        if output_file:
            try:
                self.file = open(output_file, 'w', encoding='utf-8')
                print(f"Saving output to {output_file}")
            except Exception as e:
                print(f"Warning: Could not open output file {output_file}: {e}")
                self.file = None
    
    def print(self, *args, **kwargs):
        """
        Print to both console and file if available.
        
        Args:
            *args: Variable length argument list to be printed.
            **kwargs: Arbitrary keyword arguments to be passed to print().
        """
        # Print to console
        print(*args, **kwargs)
        
        # Also write to file if available
        if self.file:
            # Convert all arguments to strings and join with spaces
            output_str = ' '.join(str(arg) for arg in args)
            
            # Handle the 'end' parameter if provided in kwargs
            end = kwargs.get('end', '\n')
            
            # Write to file
            self.file.write(output_str + end)
            self.file.flush()  # Ensure it's written immediately
    
    def close(self):
        """Close the output file if it was opened."""
        if self.file:
            self.file.close()
            self.file = None
