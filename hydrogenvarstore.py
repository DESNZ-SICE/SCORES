class cost:
    def __init__(self, low, medium, high, unit):
        """costs with low, medium and high values. The unit should also be specificed, as a string"""
        self.unit = type
        self.cost = {"low": low, "medium": medium, "high": high}


class electrolyser:
    def __init__(self, technology, capex, opex, variablecosts, efficiency, lifetime):
        self.technology = technology
        self.capex = capex
        self.opex = opex
        self.variablecosts = variablecosts
        self.efficiency = efficiency
        self.lifetime = lifetime


pemcapex = cost(954, 1159, 1888, "GBP/kW")
pemopex = cost(32.38, 36.72, 42.73, "GBP/kW/year")
pemvarcosts = cost(0.0029, 0.0039, 0.0089, "GBP/kWh")
pemefficiencies = cost(0.625, 0.71, 0.83, "unitless")

pemelectrolyer = electrolyser(
    "PEM", pemcapex, pemopex, pemvarcosts, pemefficiencies, 30
)

alkalinecapex = cost(729, 859, 1180, "GBP/kW")
alkalineopex = cost(29.89, 31.11, 36.6, "GBP/kW/year")
alkalinevarcosts = cost(0.0033, 0.0045, 0.0068, "GBP/kWh")
alkalineefficiencies = cost(0.67, 0.77, 0.83, "unitless")

alkalineelectrolyser = electrolyser(
    "Alkaline", alkalinecapex, alkalineopex, alkalinevarcosts, alkalineefficiencies, 30
)

solidoxidecapex = cost(1351, 1797, 2584, "GBP/kW")
solidoxideopex = cost(51.96, 54.77, 56.18, "GBP/kW/year")
solidoxidedvarcosts = cost(0.0079, 0.0119, 0.0201, "GBP/kWh")
solidoxideefficiencies = cost(0.714, 0.74, 0.90, "unitless")
