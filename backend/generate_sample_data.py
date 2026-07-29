"""
Generate sample particle physics dataset for testing

This creates a synthetic dataset mimicking Higgs to ZZ* decay events
"""

import pandas as pd
import numpy as np
import os

def generate_sample_dataset(n_events=10000):
    """Generate synthetic particle physics data"""
    
    np.random.seed(42)
    
    # Simulate 4-lepton events (H → ZZ* → 4l)
    data = {}
    
    # Lepton properties
    for i in range(1, 5):  # 4 leptons
        # Transverse momentum (pT) in GeV
        data[f'lepton_{i}_pt'] = np.random.exponential(30, n_events) + 10
        
        # Pseudorapidity (eta)
        data[f'lepton_{i}_eta'] = np.random.normal(0, 1.5, n_events)
        
        # Azimuthal angle (phi)
        data[f'lepton_{i}_phi'] = np.random.uniform(-np.pi, np.pi, n_events)
        
        # Energy (E) in GeV
        data[f'lepton_{i}_E'] = data[f'lepton_{i}_pt'] * np.cosh(data[f'lepton_{i}_eta'])
    
    # Z boson candidates (two pairs of leptons)
    data['Z1_mass'] = np.random.normal(91.2, 2.5, n_events)  # Z mass ~91 GeV
    data['Z2_mass'] = np.random.normal(45, 10, n_events)  # Off-shell Z*
    
    # Four-lepton invariant mass (Higgs candidate)
    # Signal events: peak around 125 GeV
    # Background: smooth distribution
    signal_fraction = 0.3
    n_signal = int(n_events * signal_fraction)
    
    mass_4l = np.zeros(n_events)
    mass_4l[:n_signal] = np.random.normal(125, 1.5, n_signal)  # Signal
    mass_4l[n_signal:] = np.random.exponential(50, n_events - n_signal) + 80  # Background
    
    # Shuffle
    np.random.shuffle(mass_4l)
    data['mass_4l'] = mass_4l
    
    # Missing transverse energy
    data['MET'] = np.random.exponential(20, n_events)
    
    # Number of jets
    data['n_jets'] = np.random.poisson(2, n_events)
    
    # Leading jet pT
    data['jet_1_pt'] = np.random.exponential(40, n_events)
    
    # Total transverse energy
    data['HT'] = np.random.gamma(5, 30, n_events)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure all values are positive where needed
    for col in df.columns:
        if 'pt' in col or 'E' in col or 'mass' in col or 'MET' in col or 'HT' in col:
            df[col] = df[col].abs()
    
    return df


def main():
    """Generate and save sample dataset"""
    
    print("🔬 Generating sample particle physics dataset...")
    
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate dataset
    df = generate_sample_dataset(n_events=10000)
    
    # Save to CSV
    output_path = 'data/raw/sample_higgs_zz.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample dataset created!")
    print(f"   Path: {output_path}")
    print(f"   Events: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print("\n📊 Sample statistics:")
    print(df.describe())
    
    print("\n💡 Next steps:")
    print("   1. Run: python add_dataset.py ../data/raw/sample_higgs_zz.csv \"Higgs ZZ Sample\" \"Sample Higgs to ZZ decay events\"")
    print("   2. Start the server: python main.py")
    print("   3. Open browser: http://localhost:8000")


if __name__ == "__main__":
    main()
