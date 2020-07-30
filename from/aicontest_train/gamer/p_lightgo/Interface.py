from __future__ import print_function

import json
import sys


# ==============================================================================
# Interfaz
# ==============================================================================


class Interface(object):

    def __init__(self, bot_class):
        self.bot_class = bot_class
        self.bot = None

    @staticmethod
    def recv():
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        return json.loads(line)

    @staticmethod
    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def run(self):
        init = self.recv()
        self.bot = self.bot_class(init)
        self.send({"name": self.bot.NAME})
        while True:
            state = self.recv()
            move = self.bot.play(state)
            self.send(move)
            status = self.recv()
            if status["success"]:
                self.bot.success()
            else:
                self.bot.error(status["message"], move)


class Bot(object):
    """Bot base. Este bot no hace nada (pasa todos los turnos)."""
    NAME = "NullBot"

    # ==========================================================================
    # Comportamiento del bot
    # Metodos a implementar / sobreescribir (opcionalmente)
    # ==========================================================================

    def __init__(self, init_state=None):
        """Inicializar el bot: llamado al comienzo del juego."""
        if init_state is not None:
            self.player_num = init_state["player_num"]
            self.player_count = init_state["player_count"]
            self.init_pos = init_state["position"]
            self.map = init_state["map"]
            self.lighthouses = list(map(tuple, init_state["lighthouses"]))
        else:
            self.player_num = -1
            self.player_count = 0
            self.init_pos = ()
            self.map = list()
            self.lighthouses = list()

    def play(self, state):
        """
        Jugar: llamado cada turno.
        Debe devolver una accion (jugada).

        state: estado actual del juego.
        """
        return self.nop()

    def success(self):
        """Exito: llamado cuando la jugada previa es valida."""
        pass

    def error(self, message, last_move):
        """Error: llamado cuando la jugada previa no es valida."""
        self.log("Recibido error: %s", message)
        self.log("Jugada previa: %r", last_move)

    # ==========================================================================
    # Utilidades
    # No es necesario sobreescribir estos metodos.
    # ==========================================================================

    def log(self, message, *args):
        """Mostrar mensaje de registro por stderr"""
        print("[%s] %s" % (self.NAME, (message % args)), file=sys.stderr)

    # ==========================================================================
    # Jugadas posibles
    # No es necesario sobreescribir estos metodos.
    # ==========================================================================

    @staticmethod
    def nop():
        """Pasar el turno"""
        return {
            "command": "pass",
        }

    @staticmethod
    def move(x, y):
        """
        Mover a una casilla adyacente

        x: delta x (0, -1, 1)
        y: delta y (0, -1, 1)
        """
        return {
            "command": "move",
            "x": x,
            "y": y
        }

    @staticmethod
    def attack(energy):
        """
        Atacar a un faro

        energy: energia (entero positivo)
        """
        return {
            "command": "attack",
            "energy": energy
        }

    @staticmethod
    def connect(destination):
        """Conectar a un faro remoto

        destination: tupla o lista (x,y): coordenadas del faro remoto
        """
        return {
            "command": "connect",
            "destination": destination
        }


if __name__ == "__main__":
    i_face = Interface(Bot)
    i_face.run()
