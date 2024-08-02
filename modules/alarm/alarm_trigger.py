import subprocess
import pulsectl

def set_sink(sink_name):
    with pulsectl.Pulse('my-client') as pulse:
        for sink in pulse.sink_list():
            if sink.name == sink_name:
                pulse.default_set(sink)

def sound_alarm():
    set_sink('alsa_output.pci-0000_00_1f.3.hdmi-stereo')
    filename = "/home/development/deepstreambackend_bagdogra/modules/alarm/alarm.wav"
    subprocess.run(["aplay",filename])

if __name__ == "__main__":
    sound_alarm()