import os
import sys
import numpy as np

# -- ensure project root is on the path --
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.channel import generate_channels

def generate_and_save_dataset(M=1000, output_file='sbl_input_dataset.npz'):
    """
    Generates a single realization of the cascaded channel and saves it 
    to a file to be used as input for SBL grouping.
    """
    print(f"Generating channel realizations for M={M} RIS elements...")
    
    # Generate random channels using existing functions
    h_BI, h_IU = generate_channels(M)
    
    # Calculate cascaded channel
    cascaded_channel = h_BI * h_IU
    
    # Calculate the features used in SBL grouping
    # (magnitude and phase of the cascaded channel)
    magnitude = np.abs(cascaded_channel)
    angle = np.angle(cascaded_channel)
    
    output_path = os.path.join(project_root, output_file)
    
    # Save all arrays into a compressed .npz file
    np.savez(output_path, 
             h_BI=h_BI, 
             h_IU=h_IU, 
             cascaded_channel=cascaded_channel,
             magnitude=magnitude,
             angle=angle)
             
    print(f"Dataset successfully saved to {output_path}")
    print("The file contains the following arrays:")
    print("  - h_BI: Base station to IRS channel")
    print("  - h_IU: IRS to User channel")
    print("  - cascaded_channel: Product of h_BI and h_IU (complex)")
    print("  - magnitude: Absolute value of cascaded_channel")
    print("  - angle: Phase angle of cascaded_channel")

if __name__ == "__main__":
    generate_and_save_dataset()
