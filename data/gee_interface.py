class GEEInterface:
    def __init__(self, project=None):
        self.project = project

    def initialize(self):
        raise NotImplementedError(
            "Google Earth Engine initialization not configured for this demo."
        )

    def fetch_flood_inundation(self, bbox, start_date, end_date):
        raise NotImplementedError(
            "Placeholder for Earth Engine flood extent query."
        )

    def fetch_chlorophyll(self, bbox, start_date, end_date):
        raise NotImplementedError(
            "Placeholder for chlorophyll-a query."
        )
