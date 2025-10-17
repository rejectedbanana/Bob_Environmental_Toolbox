"""
Functions that import JSON files exported by Tini Scientific's BoB Environmental App into Python
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Union
import math


def import_awu_data(json_file_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Import AWU JSON data and parse into location, submersion, and motion dictionaries.
    
    Args:
        json_file_path (str): Path to the JSON file exported from BoB Environmental App
        
    Returns:
        Tuple containing three dictionaries:
        - location: Dictionary with location data as numpy arrays
        - submersion: Dictionary with submersion data as numpy arrays  
        - motion: Dictionary with motion data as numpy arrays
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
        KeyError: If expected data structure is not found
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format: {e}")
    
    # Initialize dictionaries
    location = {}
    submersion = {}
    motion = {}
    
    # Parse location data
    if 'location' in data:
        location = _parse_data_section(data['location'], 'location')
    elif 'Location' in data:
        location = _parse_data_section(data['Location'], 'location')
    
    # Parse submersion data
    if 'submersion' in data:
        submersion = _parse_data_section(data['submersion'], 'submersion')
    elif 'Submersion' in data:
        submersion = _parse_data_section(data['Submersion'], 'submersion')
    
    # Parse motion data
    if 'motion' in data:
        motion = _parse_data_section(data['motion'], 'motion')
    elif 'Motion' in data:
        motion = _parse_data_section(data['Motion'], 'motion')
    
    # Automatically display import summary
    _display_import_summary(location, submersion, motion)
    
    return location, submersion, motion


def _display_import_summary(location: Dict[str, np.ndarray], 
                           submersion: Dict[str, np.ndarray], 
                           motion: Dict[str, np.ndarray]) -> None:
    """
    Display a comprehensive summary of the imported AWU data.
    
    Args:
        location: Location data dictionary
        submersion: Submersion data dictionary  
        motion: Motion data dictionary
    """
    print("✅ Successfully loaded AWU data!")
    print(f"Location data: {len(location)} fields")
    print(f"Submersion data: {len(submersion)} fields")
    print(f"Motion data: {len(motion)} fields")
    
    # Check if submersion data contains actual measurements
    submersion_has_data = False
    submersion_sample_count = 0
    if submersion:
        for key, array in submersion.items():
            if len(array) > 0:
                submersion_has_data = True
                submersion_sample_count = max(submersion_sample_count, len(array))
                break
    
    if submersion_has_data:
        print(f"🌊 Submersion data available: {submersion_sample_count} samples")
        print(f"   Available measurements: {[key for key, array in submersion.items() if len(array) > 0]}")
    else:
        print("🏄 No submersion data (device was above water)")
    
    # Check motion data availability
    motion_sample_count = 0
    if 'accelerationX' in motion:
        motion_sample_count = len(motion['accelerationX'])
    
    # Check location data availability  
    location_sample_count = 0
    if 'latitude' in location:
        location_sample_count = len(location['latitude'])
    
    print(f"\n📊 Data Summary:")
    print(f"   • Location samples: {location_sample_count}")
    print(f"   • Motion samples: {motion_sample_count}")
    print(f"   • Submersion samples: {submersion_sample_count}")
    
    # Display motion data keys
    print(f"\nLocation data fields: {list(location.keys())}")
    print(f"\nMotion data fields: {list(motion.keys())}")
    print(f"\nSubmersion data fields: {list(submersion.keys())}")


def _parse_data_section(section_data: Union[Dict, List], section_name: str) -> Dict[str, np.ndarray]:
    """
    Parse a data section and convert values to numpy arrays.
    
    Args:
        section_data: Data section from JSON (dict or list)
        section_name: Name of the section for error reporting
        
    Returns:
        Dictionary with numpy arrays as values
    """
    result = {}
    
    if isinstance(section_data, dict):
        # Check if this section has the AWU format with 'values' key
        if 'values' in section_data and isinstance(section_data['values'], dict):
            # Parse the nested values dictionary
            for key, value in section_data['values'].items():
                result[key] = _convert_to_numpy_array(value, f"{section_name}.{key}")
        else:
            # Handle flat dictionary structure
            for key, value in section_data.items():
                if key not in ['description', 'labels', 'sensor_id', 'units']:  # Skip metadata
                    result[key] = _convert_to_numpy_array(value, f"{section_name}.{key}")
    elif isinstance(section_data, list):
        # Handle list of measurements
        if section_data:
            # Assume list contains dictionaries with consistent keys
            if isinstance(section_data[0], dict):
                # Collect all unique keys
                all_keys = set()
                for item in section_data:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())
                
                # Create arrays for each key
                for key in all_keys:
                    values = []
                    for item in section_data:
                        if isinstance(item, dict) and key in item:
                            values.append(item[key])
                        else:
                            values.append(None)  # Handle missing values
                    result[key] = _convert_to_numpy_array(values, f"{section_name}.{key}")
            else:
                # List of simple values
                result['values'] = _convert_to_numpy_array(section_data, f"{section_name}.values")
    
    return result


def _convert_to_numpy_array(data: Any, field_name: str) -> np.ndarray:
    """
    Convert data to numpy array, handling various input types.
    For timestamp fields, convert to pandas datetime objects.
    
    Args:
        data: Input data to convert
        field_name: Name of the field for error reporting
        
    Returns:
        numpy array
    """
    try:
        # Special handling for timestamp fields
        if 'timestamp' in field_name.lower() and isinstance(data, (list, tuple)):
            try:
                # Convert ISO 8601 timestamps to datetime objects
                datetime_objects = pd.to_datetime(data)
                return datetime_objects.values  # Return as numpy array of datetime64
            except Exception as e:
                print(f"Warning: Could not convert timestamps in {field_name}: {e}")
                # Fall back to object array
                return np.array(data, dtype=object)
        
        if isinstance(data, (list, tuple)):
            # Handle nested lists or mixed types
            if data and isinstance(data[0], (list, tuple)):
                # 2D array case
                return np.array(data, dtype=object)
            else:
                # Try to create numeric array, fall back to object array
                try:
                    return np.array(data, dtype=float)
                except (ValueError, TypeError):
                    return np.array(data, dtype=object)
        elif isinstance(data, (int, float)):
            return np.array([data])
        elif isinstance(data, str):
            # Check if it's a single timestamp
            if 'timestamp' in field_name.lower():
                try:
                    datetime_obj = pd.to_datetime([data])
                    return datetime_obj.values
                except Exception:
                    pass
            
            # Try to parse as number, otherwise keep as string
            try:
                return np.array([float(data)])
            except ValueError:
                return np.array([data])
        else:
            return np.array([data], dtype=object)
    except Exception as e:
        print(f"Warning: Could not convert {field_name} to numpy array: {e}")
        return np.array([data], dtype=object)


def load_and_display_summary(json_file_path: str) -> None:
    """
    Load AWU data and display a summary of the imported data.
    
    Args:
        json_file_path (str): Path to the JSON file
    """
    try:
        location, submersion, motion = import_awu_data(json_file_path)
        
        print("AWU Data Import Summary")
        print("=" * 50)
        
        print(f"\nLocation data ({len(location)} fields):")
        for key, array in location.items():
            if 'timestamp' in key.lower() and len(array) > 0:
                print(f"  {key}: {array.shape} {array.dtype} (range: {array[0]} to {array[-1]})")
            else:
                print(f"  {key}: {array.shape} {array.dtype}")
            
        print(f"\nSubmersion data ({len(submersion)} fields):")
        for key, array in submersion.items():
            if 'timestamp' in key.lower() and len(array) > 0:
                print(f"  {key}: {array.shape} {array.dtype} (range: {array[0]} to {array[-1]})")
            else:
                print(f"  {key}: {array.shape} {array.dtype}")
            
        print(f"\nMotion data ({len(motion)} fields):")
        for key, array in motion.items():
            if 'timestamp' in key.lower() and len(array) > 0:
                print(f"  {key}: {array.shape} {array.dtype} (range: {array[0]} to {array[-1]})")
            else:
                print(f"  {key}: {array.shape} {array.dtype}")
            
    except Exception as e:
        print(f"Error loading data: {e}")


def example_usage():
    """
    Example of how to use the AWU importer with sample data analysis.
    """
    # Load the example data
    json_file = 'DATA/20250723_112923_Waves_AWUData.json'
    
    try:
        location, submersion, motion = import_awu_data(json_file)
        
        print("Example AWU Data Analysis")
        print("=" * 40)
        
        # Location analysis
        if 'latitude' in location and len(location['latitude']) > 0:
            print(f"\nLocation Summary:")
            print(f"  GPS points: {len(location['latitude'])}")
            print(f"  Latitude range: {location['latitude'].min():.6f} to {location['latitude'].max():.6f}")
            print(f"  Longitude range: {location['longitude'].min():.6f} to {location['longitude'].max():.6f}")
        
        # Motion analysis
        if 'accelerationX' in motion and len(motion['accelerationX']) > 0:
            print(f"\nMotion Summary:")
            print(f"  Acceleration samples: {len(motion['accelerationX'])}")
            
            total_acceleration = np.sqrt(
                motion['accelerationX']**2 + 
                motion['accelerationY']**2 + 
                motion['accelerationZ']**2
            )
            print(f"  Total acceleration - mean: {total_acceleration.mean():.3f} m/s², std: {total_acceleration.std():.3f} m/s²")
            
            # Angular velocity magnitude
            if 'angularVelocityX' in motion:
                total_angular = np.sqrt(
                    motion['angularVelocityX']**2 + 
                    motion['angularVelocityY']**2 + 
                    motion['angularVelocityZ']**2
                )
                print(f"  Angular velocity - mean: {total_angular.mean():.3f} rad/s, std: {total_angular.std():.3f} rad/s")
        
        # Submersion analysis
        if 'depth' in submersion and len(submersion['depth']) > 0:
            print(f"\nSubmersion Summary:")
            print(f"  Depth samples: {len(submersion['depth'])}")
            print(f"  Depth range: {submersion['depth'].min():.3f} to {submersion['depth'].max():.3f} meters")
            print(f"  Temperature range: {submersion['temperature'].min():.1f} to {submersion['temperature'].max():.1f} °C")
        else:
            print(f"\nSubmersion Summary: No submersion data (device was not underwater)")
            
    except FileNotFoundError:
        print(f"Example data file not found: {json_file}")
        print("Place your AWU JSON file in the DATA folder to run the example.")


# Example usage
if __name__ == "__main__":
    # Example of how to use the import function
    # Replace 'your_file.json' with the actual path to your AWU JSON file
    
    # Basic usage:
    # location_data, submersion_data, motion_data = import_awu_data('your_file.json')
    
    # Example of accessing the data:
    # print("Location keys:", list(location_data.keys()))
    # print("Submersion keys:", list(submersion_data.keys()))
    # print("Motion keys:", list(motion_data.keys()))
    
    print("AWU Importer ready. Use import_awu_data('your_file.json') to load data.")
    print("Run example_usage() to see analysis of the sample data file.")


def get_magnetic_declination(latitude: float, longitude: float, year: int = 2025) -> float:
    """
    Approximate magnetic declination for a given location and year.
    
    This is a simplified calculation. For precise scientific work, use NOAA's 
    World Magnetic Model or similar authoritative sources.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees  
        year: Year (default 2025)
        
    Returns:
        Magnetic declination in degrees (positive = east, negative = west)
    """
    # Simplified global model - for precise work, use NOAA WMM
    # This approximation is based on IGRF model coefficients
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    
    # Rough approximation for global declination pattern
    # Note: This is simplified - real calculations require spherical harmonics
    declination = (
        -5.0 * math.sin(lat_rad) * math.cos(2 * lon_rad) +
        2.0 * math.cos(lat_rad) * math.sin(lon_rad) +
        1.0 * math.sin(2 * lat_rad)
    )
    
    return declination


def calculate_gps_heading(location_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Calculate heading (bearing) from GPS track.
    
    Args:
        location_data: Dictionary containing 'latitude', 'longitude', 'timestamp'
        
    Returns:
        Array of headings in degrees (0° = North, 90° = East)
    """
    if len(location_data['latitude']) < 2:
        return np.array([0.0])  # Default to North if insufficient data
    
    lats = location_data['latitude']
    lons = location_data['longitude']
    
    # Calculate bearing between consecutive GPS points
    headings = []
    
    for i in range(len(lats) - 1):
        lat1, lon1 = math.radians(lats[i]), math.radians(lons[i])
        lat2, lon2 = math.radians(lats[i + 1]), math.radians(lons[i + 1])
        
        dlon = lon2 - lon1
        
        # Forward azimuth calculation
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.degrees(math.atan2(y, x))
        
        # Normalize to 0-360°
        bearing = (bearing + 360) % 360
        headings.append(bearing)
    
    # Add final heading (same as previous)
    headings.append(headings[-1] if headings else 0.0)
    
    return np.array(headings)


def transform_to_earth_coordinates(motion_data: Dict[str, np.ndarray], 
                                 location_data: Dict[str, np.ndarray],
                                 use_magnetic_declination: bool = True) -> Dict[str, np.ndarray]:
    """
    Transform Apple Watch motion vectors from device coordinates to Earth coordinates.
    
    Apple Watch coordinate system:
    - X: Right side of watch (when crown is on right)
    - Y: Top of watch (toward crown)  
    - Z: Out from watch face (toward user)
    
    Earth coordinate system (ENU - East/North/Up):
    - X_earth: East
    - Y_earth: North
    - Z_earth: Up (opposite to gravity)
    
    Args:
        motion_data: Dictionary with motion sensor data (acceleration, angular velocity, magnetic field)
        location_data: Dictionary with GPS data (latitude, longitude, timestamp)
        use_magnetic_declination: Whether to correct for magnetic declination
        
    Returns:
        Dictionary with transformed motion data in Earth coordinates
    """
    
    # Get GPS heading (track direction)
    gps_headings = calculate_gps_heading(location_data)
    
    # Get average location for magnetic declination
    if use_magnetic_declination and len(location_data['latitude']) > 0:
        avg_lat = np.mean(location_data['latitude'])
        avg_lon = np.mean(location_data['longitude'])
        mag_declination = get_magnetic_declination(avg_lat, avg_lon)
        print(f"Applied magnetic declination: {mag_declination:.2f}° for location ({avg_lat:.3f}, {avg_lon:.3f})")
    else:
        mag_declination = 0.0
    
    # Interpolate GPS heading to motion data timestamps if needed
    if len(motion_data['timestamp']) != len(gps_headings):
        # Simple interpolation - match timestamps
        motion_times = motion_data['timestamp'].astype('datetime64[ns]').astype(float)
        location_times = location_data['timestamp'].astype('datetime64[ns]').astype(float)
        
        # Interpolate headings to motion timestamps
        headings_interp = np.interp(motion_times, location_times, gps_headings)
    else:
        headings_interp = gps_headings
    
    # Apply magnetic declination correction
    headings_corrected = (headings_interp + mag_declination) * np.pi / 180.0  # Convert to radians
    
    # Initialize output dictionary
    earth_motion = {}
    
    # Transform acceleration vectors
    if all(key in motion_data for key in ['accelerationX', 'accelerationY', 'accelerationZ']):
        acc_x_device = motion_data['accelerationX']
        acc_y_device = motion_data['accelerationY'] 
        acc_z_device = motion_data['accelerationZ']
        
        # Rotation matrix from device to Earth coordinates
        # Assuming watch is held horizontally (face up) and aligned with GPS track
        cos_h = np.cos(headings_corrected)
        sin_h = np.sin(headings_corrected)
        
        # Transform: device Y points along GPS track, device X perpendicular to track
        # Earth coordinates: X=East, Y=North, Z=Up
        earth_motion['accelerationEast'] = acc_x_device * cos_h - acc_y_device * sin_h
        earth_motion['accelerationNorth'] = acc_x_device * sin_h + acc_y_device * cos_h
        earth_motion['accelerationUp'] = -acc_z_device  # Watch Z points toward user, Earth Z points up
    
    # Transform angular velocity vectors
    if all(key in motion_data for key in ['angularVelocityX', 'angularVelocityY', 'angularVelocityZ']):
        gyro_x_device = motion_data['angularVelocityX']
        gyro_y_device = motion_data['angularVelocityY']
        gyro_z_device = motion_data['angularVelocityZ']
        
        cos_h = np.cos(headings_corrected)
        sin_h = np.sin(headings_corrected)
        
        earth_motion['angularVelocityEast'] = gyro_x_device * cos_h - gyro_y_device * sin_h
        earth_motion['angularVelocityNorth'] = gyro_x_device * sin_h + gyro_y_device * cos_h
        earth_motion['angularVelocityUp'] = -gyro_z_device
    
    # Transform magnetic field vectors
    if all(key in motion_data for key in ['magneticFieldX', 'magneticFieldY', 'magneticFieldZ']):
        mag_x_device = motion_data['magneticFieldX']
        mag_y_device = motion_data['magneticFieldY']
        mag_z_device = motion_data['magneticFieldZ']
        
        cos_h = np.cos(headings_corrected)
        sin_h = np.sin(headings_corrected)
        
        earth_motion['magneticFieldEast'] = mag_x_device * cos_h - mag_y_device * sin_h
        earth_motion['magneticFieldNorth'] = mag_x_device * sin_h + mag_y_device * cos_h
        earth_motion['magneticFieldUp'] = -mag_z_device
    
    # Copy timestamp
    if 'timestamp' in motion_data:
        earth_motion['timestamp'] = motion_data['timestamp']
    
    return earth_motion


def analyze_wave_motion(earth_motion: Dict[str, np.ndarray], 
                       sampling_rate: float = 5.0) -> Dict[str, Any]:
    """
    Analyze wave motion characteristics from Earth-coordinate motion data.
    
    Args:
        earth_motion: Motion data in Earth coordinates (East/North/Up)
        sampling_rate: Data sampling rate in Hz
        
    Returns:
        Dictionary with wave analysis results
    """
    results = {}
    
    if 'accelerationUp' in earth_motion:
        # Vertical acceleration analysis (primary wave motion)
        acc_up = earth_motion['accelerationUp']
        
        # Remove gravity and get dynamic acceleration
        acc_dynamic = acc_up - np.mean(acc_up)
        
        # Wave height estimation (double integration of acceleration)
        # This is approximate - real wave height needs proper filtering
        dt = 1.0 / sampling_rate
        velocity = np.cumsum(acc_dynamic) * dt
        velocity = velocity - np.mean(velocity)  # Remove DC drift
        
        displacement = np.cumsum(velocity) * dt
        displacement = displacement - np.mean(displacement)
        
        results['wave_height_estimate_m'] = np.std(displacement) * 4  # Significant wave height approximation
        results['vertical_acceleration_rms'] = np.sqrt(np.mean(acc_dynamic**2))
        
        # Frequency analysis
        fft_values = np.fft.fft(acc_dynamic)
        frequencies = np.fft.fftfreq(len(acc_dynamic), dt)
        power_spectrum = np.abs(fft_values)**2
        
        # Find peak frequency (dominant wave period)
        pos_freq_idx = frequencies > 0
        peak_idx = np.argmax(power_spectrum[pos_freq_idx])
        peak_frequency = frequencies[pos_freq_idx][peak_idx]
        
        results['dominant_wave_period_s'] = 1.0 / peak_frequency if peak_frequency > 0 else np.inf
        results['dominant_wave_frequency_hz'] = peak_frequency
    
    if all(key in earth_motion for key in ['accelerationEast', 'accelerationNorth']):
        # Horizontal motion analysis
        acc_east = earth_motion['accelerationEast'] - np.mean(earth_motion['accelerationEast'])
        acc_north = earth_motion['accelerationNorth'] - np.mean(earth_motion['accelerationNorth'])
        
        horizontal_magnitude = np.sqrt(acc_east**2 + acc_north**2)
        results['horizontal_acceleration_rms'] = np.sqrt(np.mean(horizontal_magnitude**2))
        
        # Wave direction (dominant direction of horizontal motion)
        mean_east = np.mean(acc_east)
        mean_north = np.mean(acc_north)
        wave_direction = math.degrees(math.atan2(mean_east, mean_north))
        wave_direction = (wave_direction + 360) % 360  # Normalize to 0-360°
        
        results['wave_direction_deg'] = wave_direction
    
    return results

