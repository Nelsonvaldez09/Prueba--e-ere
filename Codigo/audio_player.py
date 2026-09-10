import os
import json
import pygame

class AudioNotifier:
    def __init__(self, mapper_json_filename="audio_mapper.json"):
        pygame.mixer.init()
        
        # Ruta dinámica del directorio actual
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        mapper_json_path = os.path.join(self.script_dir, mapper_json_filename)
        
        if not os.path.exists(mapper_json_path):
            raise FileNotFoundError(f"[AudioNotifier] No se encontró el mapeo: {mapper_json_path}")
            
        with open(mapper_json_path, "r", encoding="utf-8") as f:
            self.audio_map = json.load(f)

    def _rutas_candidatas(self, folder_name, filename):
        """Prueba varias ubicaciones posibles de la carpeta de audios,
        porque según cómo se haya organizado el proyecto puede estar
        directamente junto al script o dentro de 'generar_audios.bat'."""
        return [
            os.path.join(self.script_dir, folder_name, filename),
            os.path.join(self.script_dir, "generar_audios.bat", folder_name, filename),
        ]

    def play_warning(self, obj_class: str, position: str, distance: str, lang: str = "es"):
        if pygame.mixer.music.get_busy():
            return

        try:
            # Obtiene el nombre exacto del archivo accediendo al nivel del idioma [lang]
            filename = self.audio_map[obj_class][position][distance][lang]

            folder_name = f"audios_{lang}"
            candidatos = self._rutas_candidatas(folder_name, filename)
            audio_path = next((p for p in candidatos if os.path.exists(p)), None)

            if audio_path:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                print(f"Reproduciendo ({lang}): {audio_path}")
            else:
                print(f"[AudioNotifier] Archivo de audio no encontrado. Probé:")
                for p in candidatos:
                    print(f"   - {p}")

        except KeyError:
            print(f"[AudioNotifier] Parámetros no encontrados en JSON: {obj_class} | {position} | {distance} | {lang}")

if __name__ == "__main__":
    notifier = AudioNotifier()
    print("Probando módulo de audio multilingüe...")
    
    # Prueba reproduciendo apyka_frente_cerca.wav
    notifier.play_warning(obj_class="chair", position="front", distance="close", lang="gn")
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)