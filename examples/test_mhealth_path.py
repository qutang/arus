import arus.libs.mhealth_format.path as m_path

found = m_path.extract_existing_hourly_filepaths(
    "/mnt/d/data/muss_data/SPADES_1/MasterSynced/2015/09/24/15/ActigraphGT9X-AccelerationCalibrated-NA.TAS1E23150066-AccelerationCalibrated.2015-09-24-15-30-15-070-M0400.sensor.csv")

print(found)
