import abc
import pickle
import copy
import sys
import os
import time
import random
import json
import getpass 
from enum import Enum

# ==========================================
# 0. 环境检测
# ==========================================
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception:
    TKINTER_AVAILABLE = False

# ==========================================
# 1. 基础数据结构
# ==========================================

class PieceType(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def opposite(self):
        if self == PieceType.BLACK: return PieceType.WHITE
        if self == PieceType.WHITE: return PieceType.BLACK
        return PieceType.EMPTY

class GameType(Enum):
    GOMOKU = 1
    GO = 2
    REVERSI = 3

class GameResult(Enum):
    NONE = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3

class CustomError(Exception): pass
class InvalidMoveError(CustomError): pass

# ==========================================
# 2. 用户系统
# ==========================================

class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.wins = 0
        self.total_games = 0

    @property
    def losses(self):
        return self.total_games - self.wins

    def get_win_rate(self):
        if self.total_games == 0: return "0.0%"
        return f"{(self.wins / self.total_games * 100):.1f}%"

    def to_dict(self):
        return {"username": self.username, "password": self.password, "wins": self.wins, "total_games": self.total_games}

    @staticmethod
    def from_dict(data):
        u = User(data["username"], data["password"])
        u.wins = data["wins"]
        u.total_games = data["total_games"]
        return u

class UserManager:
    FILE_PATH = "users.json"
    
    def __init__(self):
        self.users = {}
        self.current_user = None
        self.load_users()

    def load_users(self):
        if os.path.exists(self.FILE_PATH):
            try:
                with open(self.FILE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for uname, udata in data.items():
                        self.users[uname] = User.from_dict(udata)
            except: self.users = {}

    def save_users(self):
        data = {uname: user.to_dict() for uname, user in self.users.items()}
        try:
            with open(self.FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except: pass

    def register(self, username, password):
        if not username or not password: return False, "不能为空"
        if username in self.users: return False, "用户已存在"
        self.users[username] = User(username, password)
        self.save_users()
        return True, "注册成功"

    def login(self, username, password):
        user = self.users.get(username)
        if user and user.password == password:
            self.current_user = user
            return True, "登录成功"
        return False, "用户名或密码错误"
    
    def get_user(self, username):
        return self.users.get(username)

    def record_game_result_by_name(self, username, is_win):
        user = self.users.get(username)
        if user:
            user.total_games += 1
            if is_win: user.wins += 1
            self.save_users()

# ==========================================
# 3. 玩家与 AI 策略
# ==========================================

class Player(abc.ABC):
    def __init__(self, color: PieceType, name="Player", is_registered=False):
        self.color = color
        self.name = name
        self.is_registered = is_registered

    @abc.abstractmethod
    def get_move(self, board, rule): pass

class HumanPlayer(Player):
    def get_move(self, board, rule): return None

class AIStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_move(self, board, rule, color): pass

class RandomStrategy(AIStrategy):
    def compute_move(self, board, rule, color):
        legal_moves = []
        for r in range(board.size):
            for c in range(board.size):
                try:
                    if rule.validate_move(board, r, c, color, silent=True):
                        legal_moves.append((r, c))
                except: continue
        return random.choice(legal_moves) if legal_moves else None

class GreedyGomokuStrategy(AIStrategy):
    def compute_move(self, board, rule, color):
        mid = board.size // 2
        if board.get_piece(mid, mid) == PieceType.EMPTY: return (mid, mid)
        opponent = color.opposite()
        best_score = -1
        best_moves = []
        
        for r in range(board.size):
            for c in range(board.size):
                if board.get_piece(r, c) != PieceType.EMPTY: continue
                score = (board.size - abs(r - mid) - abs(c - mid))
                attack = self.evaluate_line(board, r, c, color)
                defense = self.evaluate_line(board, r, c, opponent)
                score += attack * 1.0 
                score += defense * 1.1 
                if score > best_score:
                    best_score = score
                    best_moves = [(r, c)]
                elif score == best_score:
                    best_moves.append((r, c))
        
        if not best_moves: return RandomStrategy().compute_move(board, rule, color)
        return random.choice(best_moves)

    def evaluate_line(self, board, r, c, color):
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        total_val = 0
        for dr, dc in dirs:
            count = 1
            for k in range(1, 5):
                nr, nc = r + dr*k, c + dc*k
                if board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == color: count += 1
                else: break
            for k in range(1, 5):
                nr, nc = r - dr*k, c - dc*k
                if board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == color: count += 1
                else: break
            if count >= 5: total_val += 100000
            elif count == 4: total_val += 5000
            elif count == 3: total_val += 500
            elif count == 2: total_val += 50
        return total_val

class GreedyReversiStrategy(AIStrategy):
    def compute_move(self, board, rule, color):
        legal_moves = []
        best_score = -9999
        best_move = None
        for r in range(board.size):
            for c in range(board.size):
                try:
                    if rule.validate_move(board, r, c, color, silent=True):
                        flips = rule.count_flips(board, r, c, color)
                        score = flips
                        if (r, c) in [(0,0), (0,7), (7,0), (7,7)]: score += 100
                        if (r, c) in [(0,1), (1,0), (1,1), (0,6), (1,7), (1,6), (6,0), (7,1), (6,1), (6,7), (7,6), (6,6)]: score -= 20
                        if score > best_score:
                            best_score = score
                            best_move = (r, c)
                        legal_moves.append((r, c))
                except: continue
        return best_move if best_move else (random.choice(legal_moves) if legal_moves else None)

class AIPlayer(Player):
    def __init__(self, color, strategy: AIStrategy, name="AI"):
        super().__init__(color, name, is_registered=False)
        self.strategy = strategy
    def get_move(self, board, rule):
        return self.strategy.compute_move(board, rule, self.color)

# ==========================================
# 4. 模型层 (Model)
# ==========================================

class Board:
    def __init__(self, size):
        self.size = size
        self.grid = [[PieceType.EMPTY for _ in range(size)] for _ in range(size)]
    def is_within_bounds(self, r, c): return 0 <= r < self.size and 0 <= c < self.size
    def get_piece(self, r, c): return self.grid[r][c] if self.is_within_bounds(r, c) else None
    def place_piece(self, r, c, p): 
        if self.is_within_bounds(r, c): self.grid[r][c] = p
    def copy(self):
        n = Board(self.size)
        n.grid = copy.deepcopy(self.grid)
        return n

class GameStateMemento:
    def __init__(self, board, current_player_color, is_game_over, last_move=None):
        self.board = board.copy()
        self.current_player_color = current_player_color
        self.is_game_over = is_game_over
        self.last_move = last_move

# ==========================================
# 5. 规则层 (Strategy)
# ==========================================

class GameRule(abc.ABC):
    @abc.abstractmethod
    def check_win(self, board, last_move) -> GameResult: pass
    @abc.abstractmethod
    def validate_move(self, board, r, c, current_player, silent=False) -> bool: pass
    @abc.abstractmethod
    def process_move_logic(self, board, r, c, current_player): pass
    @abc.abstractmethod
    def has_legal_move(self, board, current_player) -> bool: pass

class GomokuRule(GameRule):
    def validate_move(self, board, r, c, current_player, silent=False):
        if not board.is_within_bounds(r, c): 
            if silent: return False
            raise InvalidMoveError("超出范围")
        if board.get_piece(r, c) != PieceType.EMPTY: 
            if silent: return False
            raise InvalidMoveError("已有棋子")
        return True
    def process_move_logic(self, board, r, c, current_player): pass
    def check_win(self, board, last_move) -> GameResult:
        if not last_move: return GameResult.NONE
        r, c, color = last_move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                if board.get_piece(r + dr*i, c + dc*i) == color: count += 1
                else: break
            for i in range(1, 5):
                if board.get_piece(r - dr*i, c - dc*i) == color: count += 1
                else: break
            if count >= 5: return GameResult.BLACK_WIN if color == PieceType.BLACK else GameResult.WHITE_WIN
        return GameResult.NONE
    def has_legal_move(self, board, current_player) -> bool:
        return any(cell == PieceType.EMPTY for row in board.grid for cell in row)

class GoRule(GameRule):
    def __init__(self): self.ko_point = None
    def get_liberties(self, board, r, c, color):
        stack, visited, liberties, group = [(r, c)], {(r, c)}, set(), {(r, c)}
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if not board.is_within_bounds(nx, ny): continue
                p = board.get_piece(nx, ny)
                if p == PieceType.EMPTY: liberties.add((nx, ny))
                elif p == color and (nx, ny) not in visited:
                    visited.add((nx, ny)); group.add((nx, ny)); stack.append((nx, ny))
        return group, liberties
    def validate_move(self, board, r, c, current_player, silent=False):
        if not board.is_within_bounds(r, c) or board.get_piece(r, c) != PieceType.EMPTY:
            if silent: return False
            raise InvalidMoveError("位置无效")
        if self.ko_point == (r, c):
            if silent: return False
            raise InvalidMoveError("打劫禁着")
        return True
    def process_move_logic(self, board, r, c, current_player):
        opp = current_player.opposite()
        captured = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = r+dx, c+dy
            if board.is_within_bounds(nx, ny) and board.get_piece(nx, ny) == opp:
                grp, libs = self.get_liberties(board, nx, ny, opp)
                if not libs:
                    for gx, gy in grp: board.place_piece(gx, gy, PieceType.EMPTY); captured.append((gx, gy))
        grp, libs = self.get_liberties(board, r, c, current_player)
        if not libs: raise InvalidMoveError("自杀禁着")
        if len(captured) == 1 and len(grp) == 1: self.ko_point = captured[0]
        else: self.ko_point = None
    def check_win(self, board, last_move): return GameResult.NONE
    def has_legal_move(self, board, current_player): return True 

class ReversiRule(GameRule):
    def count_flips(self, board, r, c, color):
        opp = color.opposite()
        dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        total = 0
        for dr, dc in dirs:
            cur = 0; nr, nc = r+dr, c+dc
            while board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == opp:
                nr+=dr; nc+=dc; cur+=1
            if board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == color: total += cur
        return total
    def validate_move(self, board, r, c, current_player, silent=False):
        if not board.is_within_bounds(r, c) or board.get_piece(r, c) != PieceType.EMPTY:
            if silent: return False
            raise InvalidMoveError("位置无效")
        if self.count_flips(board, r, c, current_player) == 0:
            if silent: return False
            raise InvalidMoveError("必须夹住对方")
        return True
    def process_move_logic(self, board, r, c, current_player):
        opp = current_player.opposite()
        for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            flip = []; nr, nc = r+dr, c+dc
            while board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == opp:
                flip.append((nr, nc)); nr+=dr; nc+=dc
            if board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == current_player:
                for fx, fy in flip: board.place_piece(fx, fy, current_player)
    def check_win(self, board, last_move): return GameResult.NONE
    def has_legal_move(self, board, current_player) -> bool:
        for r in range(board.size):
            for c in range(board.size):
                if self.validate_move(board, r, c, current_player, silent=True): return True
        return False

# ==========================================
# 6. 核心逻辑层
# ==========================================

class GameEngine:
    def __init__(self):
        self.board = None
        self.rule = None
        self.players = {}
        self.current_player_color = PieceType.BLACK
        self.is_game_over = True
        self.history = []
        self.observers = []
        self.user_manager = UserManager()
        self.replay_mode = False
        self.replay_index = 0
        self.tk_root = None 

    def start_game(self, game_type, size, p1_obj, p2_obj):
        self.board = Board(size)
        self.is_game_over = False
        self.history = []
        self.pass_count = 0
        self.replay_mode = False
        self.current_player_color = PieceType.BLACK
        self.game_type = game_type

        if game_type == GameType.GOMOKU: self.rule = GomokuRule()
        elif game_type == GameType.GO: self.rule = GoRule()
        elif game_type == GameType.REVERSI:
            self.rule = ReversiRule()
            mid = size // 2
            self.board.place_piece(mid-1, mid-1, PieceType.WHITE)
            self.board.place_piece(mid, mid, PieceType.WHITE)
            self.board.place_piece(mid-1, mid, PieceType.BLACK)
            self.board.place_piece(mid, mid-1, PieceType.BLACK)

        self.players[PieceType.BLACK] = p1_obj
        self.players[PieceType.WHITE] = p2_obj
        
        self.save_state()
        self.notify_observers("游戏开始")

    def ai_step(self):
        if self.is_game_over or self.replay_mode: return
        player = self.players[self.current_player_color]
        if isinstance(player, AIPlayer):
            self.notify_observers(f"{player.name} 思考中...")
            if self.tk_root: self.tk_root.update()
            
            if self.game_type == GameType.GOMOKU:
                if not self.rule.has_legal_move(self.board, self.current_player_color):
                    self.end_game_by_score("平局(棋盘满)")
                    return
                move = player.get_move(self.board, self.rule)
                if move: self.make_move(move[0], move[1])
                else: 
                    fallback = RandomStrategy().compute_move(self.board, self.rule, self.current_player_color)
                    if fallback: self.make_move(fallback[0], fallback[1])
                    else: self.end_game_by_score("棋盘已满")
            else:
                if not self.rule.has_legal_move(self.board, self.current_player_color):
                    self.handle_pass()
                    return
                move = player.get_move(self.board, self.rule)
                if move: self.make_move(move[0], move[1])
                else: self.handle_pass()

    def make_move(self, r, c):
        if self.is_game_over: return
        try:
            self.rule.validate_move(self.board, r, c, self.current_player_color)
            self.board.place_piece(r, c, self.current_player_color)
            self.rule.process_move_logic(self.board, r, c, self.current_player_color)
            
            self.pass_count = 0
            self.save_state((r, c))
            
            win = GameResult.NONE
            if self.game_type == GameType.GOMOKU:
                win = self.rule.check_win(self.board, (r, c, self.current_player_color))
            
            is_full = all(c != PieceType.EMPTY for row in self.board.grid for c in row)
            
            if win != GameResult.NONE: self.process_win(win)
            elif is_full: self.end_game_by_score("棋盘已满")
            else:
                self.current_player_color = self.current_player_color.opposite()
                self.notify_observers()
                if self.tk_root and isinstance(self.players[self.current_player_color], AIPlayer):
                    self.tk_root.after(100, self.ai_step)

        except InvalidMoveError as e:
            self.notify_observers(f"非法落子: {e}")
            raise e

    def handle_pass(self):
        if self.game_type == GameType.GOMOKU:
            self.notify_observers("错误: 五子棋规则不允许虚着")
            return
        self.pass_count += 1
        self.current_player_color = self.current_player_color.opposite()
        if self.pass_count >= 2:
            self.end_game_by_score("双方虚着/无子可落")
        else:
            self.save_state()
            self.notify_observers("虚着/Pass")
    
    def surrender(self):
        if self.is_game_over or self.replay_mode: return
        loser_color = self.current_player_color
        winner_color = loser_color.opposite()
        p_name = self.players[loser_color].name
        if winner_color == PieceType.BLACK:
            self.process_win(GameResult.BLACK_WIN, msg=f"{p_name} 认输")
        else:
            self.process_win(GameResult.WHITE_WIN, msg=f"{p_name} 认输")

    def end_game_by_score(self, reason):
        b = sum(r.count(PieceType.BLACK) for r in self.board.grid)
        w = sum(r.count(PieceType.WHITE) for r in self.board.grid)
        res = GameResult.BLACK_WIN if b > w else (GameResult.WHITE_WIN if w > b else GameResult.DRAW)
        self.process_win(res, f"{reason} (黑:{b} 白:{w})")

    def process_win(self, result, msg=""):
        self.is_game_over = True
        self.save_state()
        
        p1 = self.players[PieceType.BLACK]
        p2 = self.players[PieceType.WHITE]

        if result != GameResult.DRAW:
            is_p1_win = (result == GameResult.BLACK_WIN)
            
            if p1.is_registered:
                self.user_manager.record_game_result_by_name(p1.name, is_p1_win)
                print(f"[数据] 玩家 {p1.name} 战绩已更新")

            if p2.is_registered:
                self.user_manager.record_game_result_by_name(p2.name, not is_p1_win)
                print(f"[数据] 玩家 {p2.name} 战绩已更新")

        self.notify_observers(f"GAME_OVER:{result} {msg}")

    def save_state(self, move=None):
        self.history.append(GameStateMemento(self.board, self.current_player_color, self.is_game_over, move))

    def reset_game(self):
        self.board = None
        self.is_game_over = True

    def undo(self):
        if self.replay_mode: 
            self.notify_observers("回放模式下无法悔棋")
            return
        if len(self.history) < 2:
            self.notify_observers("无法悔棋：已是初始状态")
            return

        steps = 1
        p_next = self.players[self.current_player_color]
        p_prev = self.players[self.current_player_color.opposite()]
        if isinstance(p_next, HumanPlayer) and isinstance(p_prev, AIPlayer): steps = 2
        
        if len(self.history) <= steps:
            self.notify_observers("无法悔棋")
            return

        for _ in range(steps): self.history.pop()
        
        last = self.history[-1]
        self.board = last.board.copy()
        self.current_player_color = last.current_player_color
        self.is_game_over = last.is_game_over
        self.pass_count = 0
        self.notify_observers("已悔棋")

    def save_game(self, filename):
        p1 = self.players[PieceType.BLACK]
        p2 = self.players[PieceType.WHITE]
        data = {
            'history': [h.board.grid for h in self.history],
            'type': self.game_type,
            'size': self.board.size,
            'p1_name': p1.name,
            'p2_name': p2.name,
            'p1_is_ai': isinstance(p1, AIPlayer),
            'p2_is_ai': isinstance(p2, AIPlayer)
        }
        try:
            with open(filename, 'wb') as f: pickle.dump(data, f)
            return True, "保存成功"
        except Exception as e: return False, str(e)

    def load_game(self, filename, jump_to_end=False):
        if not os.path.exists(filename): return False, "文件不存在"
        try:
            with open(filename, 'rb') as f: data = pickle.load(f)
        except Exception: return False, "文件损坏"
        
        saved_p1 = data.get('p1_name')
        saved_p2 = data.get('p2_name')
        current_u = self.user_manager.current_user

        if current_u:
            c_name = current_u.username
            if c_name != saved_p1 and c_name != saved_p2:
                return False, f"权限不足：此存档属于 {saved_p1} 和 {saved_p2}，您({c_name})无权查看。"
        else:
            p1_reg = saved_p1 in self.user_manager.users
            p2_reg = saved_p2 in self.user_manager.users
            if p1_reg or p2_reg:
                return False, "权限不足：此存档属于注册用户，游客无权查看。请先登录。"

        self.game_type = data['type']
        self.board = Board(data['size'])
        if self.game_type == GameType.GOMOKU: self.rule = GomokuRule()
        elif self.game_type == GameType.GO: self.rule = GoRule()
        elif self.game_type == GameType.REVERSI: self.rule = ReversiRule()

        p1_is_ai = data.get('p1_is_ai', False)
        p2_is_ai = data.get('p2_is_ai', False)
        
        def make_player(color, name, is_ai):
            if not is_ai: return HumanPlayer(color, name)
            if self.game_type == GameType.GOMOKU: strat = GreedyGomokuStrategy()
            elif self.game_type == GameType.REVERSI: strat = GreedyReversiStrategy()
            else: strat = RandomStrategy()
            return AIPlayer(color, strat, name)

        self.players = {
            PieceType.BLACK: make_player(PieceType.BLACK, data.get('p1_name', 'P1'), p1_is_ai),
            PieceType.WHITE: make_player(PieceType.WHITE, data.get('p2_name', 'P2'), p2_is_ai)
        }
        
        self.replay_data = data['history']
        self.replay_mode = True
        self.replay_index = len(self.replay_data) - 1 if jump_to_end else 0
        self.update_replay_board()
        return True, "读取成功 (输入 'resume' 可接管继续游戏)"

    def step_replay(self, direction):
        if not self.replay_mode: return
        n = self.replay_index + direction
        if 0 <= n < len(self.replay_data):
            self.replay_index = n
            self.update_replay_board()

    def update_replay_board(self):
        self.board.grid = self.replay_data[self.replay_index]
        status = "最终局面" if self.replay_index == len(self.replay_data)-1 else f"步骤 {self.replay_index}"
        self.notify_observers(f"回放: {status} (输入 resume 继续)")

    def resume_game(self):
        if not self.replay_mode:
            self.notify_observers("当前不在回放模式")
            return
        self.replay_mode = False
        self.is_game_over = False
        
        b_count = sum(r.count(PieceType.BLACK) for r in self.board.grid)
        w_count = sum(r.count(PieceType.WHITE) for r in self.board.grid)
        if b_count == w_count: self.current_player_color = PieceType.BLACK
        else: self.current_player_color = PieceType.WHITE
            
        self.history = []
        for i in range(self.replay_index + 1):
            mem = GameStateMemento(Board(self.board.size), PieceType.BLACK, False) 
            mem.board.grid = self.replay_data[i]
            self.history.append(mem)
            
        self.notify_observers("已接管比赛！请继续下棋...")
        curr = self.players[self.current_player_color]
        if isinstance(curr, AIPlayer):
             self.notify_observers(f"轮到 {curr.name} (AI)")
             if self.tk_root: self.tk_root.after(500, self.ai_step)

    def add_observer(self, o): self.observers.append(o)
    def notify_observers(self, m=None): 
        for o in self.observers: o.update_view(self, m)

# ==========================================
# 7. 视图层 (Console)
# ==========================================

class ConsoleUI:
    def __init__(self, engine):
        self.engine = engine
        self.engine.add_observer(self)
        self.game_running = False 

    def render_board(self, board):
        print("\n   " + " ".join([f"{i:2}" for i in range(board.size)]))
        for r in range(board.size):
            line = [f"{r:2} "]
            for c in range(board.size):
                p = board.get_piece(r, c)
                s = "●" if p == PieceType.BLACK else ("○" if p == PieceType.WHITE else "+")
                line.append(f" {s}")
            print(" ".join(line))
        print("")

    def update_view(self, model, message=None):
        if message:
            print(f">>> [系统] {str(message).replace('GAME_OVER:', '')}")
            if "GAME_OVER" in str(message):
                self.game_running = False
                print("\n(游戏结束，按回车返回菜单)")

        if model.board and self.game_running:
            curr = model.players.get(model.current_player_color)
            should_render = (isinstance(curr, HumanPlayer) or model.replay_mode or model.is_game_over)
            p1_ai = isinstance(model.players[PieceType.BLACK], AIPlayer)
            p2_ai = isinstance(model.players[PieceType.WHITE], AIPlayer)
            if p1_ai and p2_ai: should_render = True

            if should_render:
                self.render_board(model.board)
                if not model.replay_mode and not model.is_game_over:
                    u_obj = self.engine.user_manager.get_user(curr.name)
                    stats = f" [胜:{u_obj.wins}/总:{u_obj.total_games}]" if u_obj else " [游客]"
                    print(f"当前回合: {curr.name}{stats} ({'黑' if curr.color==PieceType.BLACK else '白'})")

    def show_profile(self):
        u = self.engine.user_manager.current_user
        if not u:
            print(">>> 当前未登录")
            return
        print("-" * 30)
        print(f"用户档案: {u.username}")
        print(f"总场次  : {u.total_games}")
        print(f"胜场    : {u.wins}")
        print(f"败场    : {u.losses}")
        print(f"胜率    : {u.get_win_rate()}")
        # [修改] 保护密码显示
        print(f"密码    : {'*' * len(u.password)} (已隐藏)")
        print(f"存档位置: {os.path.abspath('users.json')}")
        print("-" * 30)
        input("按回车继续...")

    def create_player(self, color, label, game_type):
        while True:
            print(f"\n设置 {label} ({'黑' if color==PieceType.BLACK else '白'}):")
            print("1. 人类玩家  2. AI玩家")
            c = input("选择: ").strip()
            
            if c == '1':
                if color == PieceType.BLACK and self.engine.user_manager.current_user:
                     name = self.engine.user_manager.current_user.username
                     return HumanPlayer(color, name, is_registered=True)
                
                print(f"   该玩家是否登录账号? (y/n)")
                is_login = input("   选择: ").lower()
                if is_login == 'y':
                    u_temp = input("   用户名: ")
                    # [修正] 强制使用 getpass 隐藏密码
                    p_temp = getpass.getpass("   密码 (输入时不可见): ")

                    user_obj = self.engine.user_manager.get_user(u_temp)
                    if user_obj and user_obj.password == p_temp:
                        print(f"   >>> {u_temp} 验证通过")
                        return HumanPlayer(color, u_temp, is_registered=True)
                    else:
                        print("   >>> 验证失败，将作为游客加入")
                        return HumanPlayer(color, f"Guest_{color.name}", is_registered=False)
                else:
                    return HumanPlayer(color, f"Guest_{color.name}", is_registered=False)

            elif c == '2':
                print("   AI等级: 1. 随机(弱)  2. 贪心(较强)")
                lvl = input("   选择: ").strip()
                if lvl == '2':
                    if game_type == GameType.GOMOKU: strat = GreedyGomokuStrategy()
                    elif game_type == GameType.REVERSI: strat = GreedyReversiStrategy()
                    else: strat = RandomStrategy()
                else: strat = RandomStrategy()
                return AIPlayer(color, strat, f"AI_{color.name}")

    def print_help(self):
        print("-" * 40)
        print("指令列表:")
        print(" move <r> <c> : 落子 (行 列, 从0开始)")
        print(" undo         : 悔棋")
        print(" resign       : 认输 (本局直接判负)")
        print(" pass         : 虚着 (五子棋不可用)")
        print(" save <file>  : 保存并返回菜单")
        print(" quit         : 强制退出回菜单")
        print("-" * 40)

    def start_game_flow(self):
        print("\n选择游戏: 1.五子棋  2.围棋  3.黑白棋")
        c = input(">> ").strip()
        g_map = {'1': GameType.GOMOKU, '2': GameType.GO, '3': GameType.REVERSI}
        g_type = g_map.get(c)
        if not g_type: return
        size = 15
        if g_type == GameType.GO: size = 19
        if g_type == GameType.REVERSI: size = 8
        p1 = self.create_player(PieceType.BLACK, "玩家1", g_type)
        p2 = self.create_player(PieceType.WHITE, "玩家2", g_type)
        self.game_running = True
        self.engine.start_game(g_type, size, p1, p2)

    def load_game_flow(self):
        path = input("存档文件名: ").strip()
        print("模式: 1. 一步步回放  2. 直接看最终结果")
        mode = input(">> ").strip()
        jump = (mode == '2')
        succ, msg = self.engine.load_game(path, jump_to_end=jump)
        print(msg)
        if succ:
            self.game_running = True
            self.render_board(self.engine.board)

    def main_loop(self):
        print("=== 棋类平台 CLI v7.0 (Final) ===")
        while True:
            print("\n1. 登录  2. 注册  3. 游客")
            c = input(">> ").strip()
            if c == '3': break
            
            if c == '1' or c == '2':
                u = input("用户: ")
                # [修正] 强制使用 getpass 隐藏密码
                p = getpass.getpass("密码 (输入时不可见): ")

                if c == '1':
                    if self.engine.user_manager.login(u, p)[0]: break
                elif c == '2':
                    print(self.engine.user_manager.register(u, p)[1])
        
        print(f"\n欢迎, {self.engine.user_manager.current_user.username if self.engine.user_manager.current_user else 'Guest'}")

        while True:
            if not self.game_running:
                print("\n[主菜单] 1.新游戏  2.读档  3.退出  4.查看个人信息")
                cmd = input(">> ").strip()
                if cmd == '1': self.start_game_flow()
                elif cmd == '2': self.load_game_flow()
                elif cmd == '3': break
                elif cmd == '4': self.show_profile()
            else:
                if not self.engine.is_game_over and not self.engine.replay_mode:
                    curr_p = self.engine.players[self.engine.current_player_color]
                    if isinstance(curr_p, AIPlayer):
                        time.sleep(0.5) 
                        try:
                            self.engine.ai_step()
                        except Exception as e:
                            print(f"AI出错: {e}")
                            self.game_running = False
                        continue 

                try:
                    prompt = "(replay) >> " if self.engine.replay_mode else "(game) [输入 help 查看指令] >> "
                    raw = input(prompt).strip().split()
                except KeyboardInterrupt:
                    print("\n强制退出游戏")
                    self.game_running = False
                    continue

                if not raw: continue
                op = raw[0].lower()
                
                if op == "move":
                    if len(raw) == 3:
                        try: self.engine.make_move(int(raw[1]), int(raw[2]))
                        except Exception as e: print(f"❌ {e}")
                    else: print("❌ 格式错误: move <row> <col>")
                elif op == "pass": self.engine.handle_pass()
                elif op == "undo": self.engine.undo()
                elif op == "resign": self.engine.surrender() 
                elif op == "resume": self.engine.resume_game()
                elif op == "save":
                    if len(raw) > 1:
                        self.engine.save_game(raw[1])
                        self.engine.reset_game()
                        self.game_running = False
                    else: print("❌ 请指定文件名: save <filename>")
                elif op == "quit":
                    self.engine.reset_game()
                    self.game_running = False
                elif op == "next" and self.engine.replay_mode: self.engine.step_replay(1)
                elif op == "prev" and self.engine.replay_mode: self.engine.step_replay(-1)
                elif op == "help": self.print_help() 
                else:
                    print(f"❌ 无效指令 '{op}'。输入 'help' 查看可用列表。")

class GraphicalUI:
    def __init__(self, engine):
        self.engine = engine
        print("请使用控制台模式 (ConsoleUI)")
        sys.exit()

if __name__ == "__main__":
    engine = GameEngine()
    ui = ConsoleUI(engine)
    ui.main_loop()