from scapy.all import *

def showPacket(packet):
    print(packet.show())

def sniffing(filter):
    # count가 0이면 실시간
    sniff(filter = filter, prn = showPacket, count = 0)

if __name__ == '__main__':
    filter = 'ip and (tcp or udp or icmp)'
    sniffing(filter)