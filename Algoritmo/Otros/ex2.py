
from wifi import Cell

for cell in Cell.all('wlp2s0'):
	print(cell.ssid)