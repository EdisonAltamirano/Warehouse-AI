#!/bin/python

from pyModbusTCP.server import ModbusServer, DataBank
from time import sleep
from random import uniform
# Create an instance of ModbusServer
server = ModbusServer("10.22.240.51", 12345, no_block=True)

try:
    print("Start server...")
    server.start()
    print("Server is online")
    state = [0]
    while True:
        #DataBank.set_words(0, [int(uniform(0, 100))])
        if state != DataBank.get_words(0):
            state = DataBank.get_words(0,6)
            print("Value of Registers has changed to " +str(state))
        sleep(0.5)

except:
    print("Shutdown server ...")
    server.stop()
    print("Server is offline")