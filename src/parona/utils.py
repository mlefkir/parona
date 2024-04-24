def energy2nustarchannel(energy_keV):
    return (energy_keV - 1.6)/0.04


def nustarchannel2energy(channel_nb):
    return channel_nb*0.04+1.6
