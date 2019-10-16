#Algorithm used to measure the RSSI from all near AP's


#sudo iwlist wlp2s0 scanning | egrep 'Cell |Encryption|Quality|Last beacon|ESSID'

import rssi 

"""
This class helps us scan for all available access points, within reach. 
User must supply a network interface name, at initialization. 
i.e wlan0, docker0, wlp1s0
"""

interface = 'wlp2s0'
rssi_scanner = rssi.RSSI_Scan(interface)
ssids = ['Optronica']

ap_info = rssi.getAPinfo(networks=ssids, sudo=True)
rssi_values = [ap['signal'] for ap in ap_info]
print(rssi_values)




