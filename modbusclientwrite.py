from pyModbusTCP.client import ModbusClient
import time

SERVER_HOST = "10.22.240.51"
SERVER_PORT = 12345

c = ModbusClient()

# uncomment this line to see debug message
#c.debug(True)

# define modbus server host, port
c.host(SERVER_HOST)
c.port(SERVER_PORT)

toggle = True

while True:
    # open or reconnect TCP to server
    if not c.is_open():
        if not c.open():
            print("unable to connect to "+SERVER_HOST+":"+str(SERVER_PORT))

    # if open() is ok, write coils (modbus function 0x01)
    if c.is_open():
        # write 4 bits in modbus address 0 to 3
        print("")
        print("write bits")
        print("----------")
        print("")
        #for addr in range(4):
        is_ok = c.write_multiple_registers(0, [1])
        if is_ok:
            print("bit #" + str(0) + ": write to " + str(toggle))
        else:
            print("bit #" + str(0) + ": unable to write " + str(toggle))
        time.sleep(0.5)
        '''
        time.sleep(1)
        print("")
        print("read bits")
        print("---------")
        print("")
        bits = c.read_holding_registers(0, 4)
        if bits:
            print("bits #0 to 3: "+str(bits))
        else:
            print("unable to read")
        '''
    toggle = not toggle
    # sleep 2s before next polling
    time.sleep(2)