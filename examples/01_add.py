# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************


import asnumpy as ap
import numpy as np
import time

def main():
    array_size = (1000, 1000)
    comparison_tolerance = {'rtol': 1e-5, 'atol': 1e-8}


    print(f"Generating two random arrays of size {array_size}...")
    np_a = np.random.rand(*array_size).astype(np.float32)
    np_b = np.random.rand(*array_size).astype(np.float32)
    
    ap_a = ap.ndarray.from_numpy(np_a)
    ap_b = ap.ndarray.from_numpy(np_b)


    print("Performing addition with NumPy (CPU)...")
    start_np = time.time()
    np_result = np.add(np_a, np_b)
    end_np = time.time()
    
    print("Performing addition with asnumpy (GPU/Custom Device)...")
    start_ap = time.time()
    ap_result = ap.add(ap_a, ap_b)
    np_ap_result = ap_result.to_numpy()
    end_ap = time.time()
    

    print("\n--- Verification ---")
    is_close = np.allclose(np_result, np_ap_result, 
                           rtol=comparison_tolerance['rtol'], 
                           atol=comparison_tolerance['atol'])
    
    if is_close:
        print("✅ SUCCESS: The results from numpy and asnumpy are mathematically equivalent (np.allclose is True).")
        print(f"   (Tolerance: rtol={comparison_tolerance['rtol']}, atol={comparison_tolerance['atol']})")
    else:
        print("❌ FAILURE: The results from numpy and asnumpy DO NOT match!")

    print("\n--- Performance (For Reference) ---")
    print(f"NumPy Time:     {end_np - start_np:.6f} seconds")
    print(f"asnumpy Time:   {end_ap - start_ap:.6f} seconds (Note: Time includes device-host transfer)")

if __name__ == "__main__":
    main()