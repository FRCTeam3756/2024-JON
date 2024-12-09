import json
import logging

logging.basicConfig(
    level=logging.INFO, 
    filename="logs/decision_matrix.log", 
    filemode="w"
)

################################################

class DecisionMatrix:
    def __init__(self) -> None:
        try:
            with open("decision_engine/weights.json", "r") as config_file:
                config = json.load(config_file)
                self.confidence_weight = config.get("confidence_weight", 1.0)
                self.distance_weight = config.get("distance_weight", 1.0)
                self.angle_weight = config.get("angle_weight", 1.0)  
        except FileNotFoundError:
            logging.error("Error: 'weights.json' file not found.")
            exit()
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def compute_best_game_piece(self, *game_pieces):
        if not game_pieces:
            logging.info("No game pieces provided.")
            return None
        
        if len(game_pieces) == 1:
            game_piece = game_pieces[0]
            if all(hasattr(game_piece, attr) for attr in ['confidence', 'distance', 'angle']):
                return game_piece
            else:
                missing_attributes = [attr for attr in ['confidence', 'distance', 'angle'] if not hasattr(game_piece, attr)]
                logging.error(f"Error: Game piece {game_piece} is missing attributes: {', '.join(missing_attributes)}.")
                return None
        
        highest_score = -1
        best_game_piece = None
        
        for game_piece in game_pieces:
            missing_attributes = [attr for attr in ['confidence', 'distance', 'angle'] if not hasattr(game_piece, attr)]
            if missing_attributes:
                logging.error(f"Error: Game piece {game_piece} is missing attributes: {', '.join(missing_attributes)}.")
                continue  
            
            score = (self.confidence_weight * game_piece.confidence +
                    self.distance_weight * ((120 - game_piece.distance) / 120) +
                    self.angle_weight * (1 - abs(game_piece.angle) / 180))
            
            if score > highest_score:
                highest_score = score
                best_game_piece = game_piece
        
        return best_game_piece
