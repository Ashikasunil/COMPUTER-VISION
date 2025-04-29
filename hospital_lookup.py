pincode_to_district = {
    "600001": "Chennai", "641001": "Coimbatore"
}
district_to_hospitals = {
    "Chennai": ["Apollo Cancer Centre", "MIOT International"],
    "Coimbatore": ["KMCH", "PSG Hospitals"]
}
def get_hospitals_by_pincode(pincode: str):
    district = pincode_to_district.get(pincode.strip())
    return district_to_hospitals.get(district, ["District General Hospital"]) if district else ["District General Hospital"]
