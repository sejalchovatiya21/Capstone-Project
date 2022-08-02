from smbus2 import SMBus
from mlx90614 import MLX90614

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)

amb_temp = sensor.get_amb_temp()
amb_temp_two = "{:.2f}".format(amb_temp)

obj_temp = sensor.get_obj_temp()
obj_temp_two = "{:.2f}".format(obj_temp)

print("Ambient Temperature :", amb_temp_two)
print("Object Temperature :", obj_temp_two)

bus.close()


