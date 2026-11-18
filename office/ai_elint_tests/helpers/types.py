from enum import StrEnum

radar_map = {
    "PSR1" : 1104,
	"PSR2": 1105,
	"PSR3": 1106,
	"PSR4": 1107,
    "PSR5m1": 1108,
	"PSR5m2": 1116,
	"MF03": 1113,
	"MF04": 1109,
	"MF05": 1115,
}

class RadarMapper(StrEnum):
	PSR1 = "1104"
	PSR2 = "1105"
	PSR3 = "1106"
	PSR4 = "1107"
	PSR5m1 = "1108"
	PSR5m2 = "1116"
	MF03 = "1113"
	MF04 = "1109"
	MF05 = "1115"


radar_list = [1104, 1105, 1106, 1107, 1108, 1116, 1113, 1109, 1115]
