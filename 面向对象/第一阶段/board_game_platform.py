import abc
import pickle
import copy
import sys
import os
from enum import Enum

# 尝试导入 Tkinter
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
except Exception:
    TKINTER_AVAILABLE = False

# ==========================================
# 1. 基础数据结构与枚举
# ==========================================

class PieceType(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

class GameType(Enum):
    GOMOKU = 1
    GO = 2

class GameResult(Enum):
    NONE = 0
    BLACK_WIN = 1
    WHITE_WIN = 2
    DRAW = 3

    def __str__(self):
        if self == GameResult.BLACK_WIN: return "黑方获胜"
        if self == GameResult.WHITE_WIN: return "白方获胜"
        if self == GameResult.DRAW: return "平局"
        return "进行中"

class CustomError(Exception):
    """自定义异常基类"""
    pass

class InvalidMoveError(CustomError):
    """落子位置不合法"""
    pass

class GameLogicError(CustomError):
    """逻辑错误"""
    pass

# ==========================================
# 2. 模型层 (Model)
# ==========================================

class Board:
    """棋盘类"""
    def __init__(self, size):
        self.size = size
        self.grid = [[PieceType.EMPTY for _ in range(size)] for _ in range(size)]

    def is_within_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def get_piece(self, r, c):
        if self.is_within_bounds(r, c):
            return self.grid[r][c]
        return None

    def place_piece(self, r, c, piece_type):
        if self.is_within_bounds(r, c):
            self.grid[r][c] = piece_type

    def copy(self):
        new_board = Board(self.size)
        new_board.grid = copy.deepcopy(self.grid)
        return new_board

class GameStateMemento:
    """备忘录模式：保存游戏状态快照"""
    def __init__(self, board, current_player, is_game_over):
        self.board = board.copy()
        self.current_player = current_player
        self.is_game_over = is_game_over

# ==========================================
# 3. 策略层 (Strategy) - 规则引擎
# ==========================================

class GameRule(abc.ABC):
    """抽象策略类"""
    @abc.abstractmethod
    def check_win(self, board, last_move) -> GameResult:
        pass

    @abc.abstractmethod
    def validate_move(self, board, r, c, current_player) -> bool:
        pass

    @abc.abstractmethod
    def process_move_logic(self, board, r, c, current_player):
        pass

class GomokuRule(GameRule):
    """五子棋规则"""
    def validate_move(self, board, r, c, current_player):
        if not board.is_within_bounds(r, c):
            raise InvalidMoveError("坐标超出棋盘范围")
        if board.get_piece(r, c) != PieceType.EMPTY:
            raise InvalidMoveError("该位置已有棋子")
        return True

    def process_move_logic(self, board, r, c, current_player):
        pass

    def check_win(self, board, last_move) -> GameResult:
        if not last_move:
            return GameResult.NONE
        
        r, c, color = last_move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # 正向
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if board.get_piece(nr, nc) == color:
                    count += 1
                else:
                    break
            # 反向
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if board.get_piece(nr, nc) == color:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return GameResult.BLACK_WIN if color == PieceType.BLACK else GameResult.WHITE_WIN
        
        is_full = all(cell != PieceType.EMPTY for row in board.grid for cell in row)
        return GameResult.DRAW if is_full else GameResult.NONE

class GoRule(GameRule):
    """围棋规则"""
    def __init__(self):
        self.ko_point = None 

    def get_group_liberties(self, board, r, c, target_color=None):
        if target_color is None:
            target_color = board.get_piece(r, c)
        if target_color == PieceType.EMPTY:
            return set(), set()

        stack = [(r, c)]
        visited = {(r, c)}
        liberties = set()
        group = {(r, c)}

        while stack:
            curr_r, curr_c = stack.pop()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if not board.is_within_bounds(nr, nc):
                    continue
                neighbor_piece = board.get_piece(nr, nc)
                if neighbor_piece == PieceType.EMPTY:
                    liberties.add((nr, nc))
                elif neighbor_piece == target_color and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    group.add((nr, nc))
                    stack.append((nr, nc))
        return group, liberties

    def validate_move(self, board, r, c, current_player):
        if not board.is_within_bounds(r, c):
            raise InvalidMoveError("坐标超出棋盘范围")
        if board.get_piece(r, c) != PieceType.EMPTY:
            raise InvalidMoveError("该位置已有棋子")
        if self.ko_point == (r, c):
            raise InvalidMoveError("禁着点：打劫")
        return True

    def process_move_logic(self, board, r, c, current_player):
        opponent = PieceType.WHITE if current_player == PieceType.BLACK else PieceType.BLACK
        captured_stones = []
        
        # 提对方子
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if board.is_within_bounds(nr, nc) and board.get_piece(nr, nc) == opponent:
                group, liberties = self.get_group_liberties(board, nr, nc)
                if len(liberties) == 0:
                    for stone_r, stone_c in group:
                        board.place_piece(stone_r, stone_c, PieceType.EMPTY)
                        captured_stones.append((stone_r, stone_c))

        # 检查自杀
        group, liberties = self.get_group_liberties(board, r, c, current_player)
        if len(liberties) == 0:
            raise InvalidMoveError("禁着点：自杀")

        # 更新劫材
        if len(captured_stones) == 1 and len(group) == 1:
             self.ko_point = captured_stones[0]
        else:
            self.ko_point = None

    def check_win(self, board, last_move) -> GameResult:
        is_full = all(cell != PieceType.EMPTY for row in board.grid for cell in row)
        return GameResult.DRAW if is_full else GameResult.NONE

    def count_score(self, board):
        black_score = 0
        white_score = 0
        for r in range(board.size):
            for c in range(board.size):
                p = board.get_piece(r, c)
                if p == PieceType.BLACK: black_score += 1
                elif p == PieceType.WHITE: white_score += 1
        return black_score, white_score

# ==========================================
# 4. 核心逻辑层 (Core Logic)
# ==========================================

class GameEngine:
    def __init__(self):
        self.board = None
        self.rule = None
        self.current_player = PieceType.BLACK
        self.is_game_over = False 
        self.history = []
        self.observers = []
        self.game_type = None
        self.pass_count = 0

    def start_game(self, game_type: GameType, size: int):
        if not (8 <= size <= 19):
            raise CustomError("棋盘大小必须在 8 到 19 之间")
        
        self.board = Board(size)
        self.game_type = game_type
        self.current_player = PieceType.BLACK
        self.is_game_over = False
        self.history = []
        self.pass_count = 0
        
        if game_type == GameType.GOMOKU:
            self.rule = GomokuRule()
        else:
            self.rule = GoRule()
            
        self.save_state()
        self.notify_observers("游戏开始")

    def save_state(self):
        memento = GameStateMemento(self.board, self.current_player, self.is_game_over)
        self.history.append(memento)

    def undo(self):
        if len(self.history) < 2:
            raise GameLogicError("无法悔棋：已是初始状态")
        
        self.history.pop()
        last_state = self.history[-1]
        
        self.board = last_state.board.copy()
        self.current_player = last_state.current_player
        self.is_game_over = last_state.is_game_over
        self.pass_count = 0 
        self.notify_observers("已悔棋")

    def reset_game(self):
        """重置游戏状态，准备回到菜单"""
        self.board = None
        self.is_game_over = False
        self.history = []

    def make_move(self, r, c):
        if self.is_game_over:
            raise GameLogicError("游戏已结束，请重新开始")

        if self.board.grid[r][c] != PieceType.EMPTY:
             raise InvalidMoveError("此处已有棋子")

        self.rule.validate_move(self.board, r, c, self.current_player)
        
        backup_board = self.board.copy()
        try:
            self.board.place_piece(r, c, self.current_player)
            self.rule.process_move_logic(self.board, r, c, self.current_player)
        except InvalidMoveError as e:
            self.board = backup_board
            raise e

        self.pass_count = 0
        result = self.rule.check_win(self.board, (r, c, self.current_player))
        
        if result != GameResult.NONE:
            self.is_game_over = True
            self.save_state()
            # 通知获胜信息
            self.notify_observers(f"GAME_OVER:{result}")
            return result
        
        self.switch_player()
        self.save_state()
        self.notify_observers()
        return GameResult.NONE

    def pass_turn(self):
        if self.is_game_over:
             raise GameLogicError("游戏已结束")
        if self.game_type != GameType.GO:
            raise InvalidMoveError("仅围棋支持虚着")
        
        self.pass_count += 1
        self.switch_player()
        
        if self.pass_count >= 2:
            self.is_game_over = True
            b, w = self.rule.count_score(self.board)
            winner = GameResult.BLACK_WIN if b > w else (GameResult.WHITE_WIN if w > b else GameResult.DRAW)
            res_str = f"GAME_OVER:双方虚着，游戏结束。黑子: {b}, 白子: {w}。胜者: {winner}"
            self.save_state()
            self.notify_observers(res_str)
            return winner
        
        self.save_state()
        self.notify_observers("玩家虚着")
        return GameResult.NONE

    def switch_player(self):
        self.current_player = PieceType.WHITE if self.current_player == PieceType.BLACK else PieceType.BLACK

    def save_game(self, filename):
        try:
            data = {
                'board': self.board,
                'current_player': self.current_player,
                'game_type': self.game_type,
                'history': self.history,
                'is_game_over': self.is_game_over
            }
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            raise CustomError(f"保存失败: {str(e)}")

    def load_game(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.board = data['board']
            self.current_player = data['current_player']
            self.game_type = data['game_type']
            self.history = data['history']
            self.is_game_over = data.get('is_game_over', False)
            
            if self.game_type == GameType.GOMOKU:
                self.rule = GomokuRule()
            else:
                self.rule = GoRule()
            
            self.notify_observers("读取存档成功")
        except Exception as e:
            raise CustomError(f"读取失败: {str(e)}")

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message=None):
        for obs in self.observers:
            obs.update_view(self, message)

# ==========================================
# 5. 视图层 (View)
# ==========================================

class IGameView(abc.ABC):
    @abc.abstractmethod
    def update_view(self, model: GameEngine, message: str = None):
        pass
    @abc.abstractmethod
    def start(self):
        pass

class ConsoleUI(IGameView):
    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.engine.add_observer(self)
        self.showing_help = True

    def render_board(self, board):
        print("   " + " ".join([f"{i:2}" for i in range(board.size)]))
        for r in range(board.size):
            line = [f"{r:2} "]
            for c in range(board.size):
                p = board.get_piece(r, c)
                if p == PieceType.BLACK: symbol = "●"
                elif p == PieceType.WHITE: symbol = "○"
                else: symbol = "+" 
                line.append(f" {symbol}")
            print(" ".join(line))

    def update_view(self, model, message=None):
        is_end_message = message and "GAME_OVER" in str(message)
        
        # 1. 如果有消息，先打印消息
        if message:
            clean_msg = str(message).replace("GAME_OVER:", "")
            prefix = "🏆 最终战报" if is_end_message else "系统提示"
            print(f"\n[>>> {prefix}] {clean_msg}")
        
        # 2. 打印棋盘（只要棋盘存在）
        if model.board:
            status = "已结束" if model.is_game_over else ("黑方 (●)" if model.current_player == PieceType.BLACK else "白方 (○)")
            print(f"\n=== 游戏状态: {status} ===")
            self.render_board(model.board)

        # 3. 如果游戏结束，打印“返回菜单”的提示，并显示菜单
        if model.is_game_over:
            print("\n" + "="*40)
            print("   游戏结束！正在跳转回主菜单...")
            print("="*40 + "\n")
            model.reset_game() # 逻辑复位，防止下次move误操作
            self.print_help() # 重新显示菜单

    def print_help(self):
        print("-" * 40)
        print("【主菜单】指令列表:")
        print(" start <1/2> <size>  : 开始游戏 (1=五子棋, 2=围棋)")
        print("                       示例: start 1 15")
        print(" move <r> <c>        : 落子 (行 列)")
        print(" undo                : 悔棋")
        print(" pass                : 虚着 (围棋)")
        print(" save <path>         : 保存存档")
        print(" load <path>         : 读取存档")
        print(" quit                : 退出系统")
        print("-" * 40)

    def start(self):
        print("=== 通用棋类对战平台 (控制台版) ===")
        self.print_help()
        while True:
            try:
                cmd = input(">> ").strip().split()
                if not cmd: continue
                op = cmd[0].lower()

                if op == "quit": break
                elif op == "help": self.print_help()
                elif op == "start":
                    if len(cmd) != 3: raise CustomError("参数错误")
                    g_type = GameType.GOMOKU if cmd[1] == '1' else GameType.GO
                    self.engine.start_game(g_type, int(cmd[2]))
                elif op == "move":
                    if not self.engine.board: 
                        print("错误: 游戏未开始，请输入 start 指令开始新游戏")
                        continue
                    self.engine.make_move(int(cmd[1]), int(cmd[2]))
                elif op == "pass": self.engine.pass_turn()
                elif op == "undo": self.engine.undo()
                elif op == "save": self.engine.save_game(cmd[1])
                elif op == "load": self.engine.load_game(cmd[1])
                else: print("未知指令，输入 help 查看菜单")

            except Exception as e:
                print(f"操作失败: {e}")

class GraphicalUI(IGameView):
    def __init__(self, engine: GameEngine):
        if not TKINTER_AVAILABLE:
            raise ImportError("Tkinter 模块不可用或无显示环境")
        try:
            self.root = tk.Tk()
        except Exception as e:
            raise RuntimeError(f"无法启动图形界面: {e}")

        self.engine = engine
        self.engine.add_observer(self)
        self.root.title("通用棋类对战平台")
        self.cell_size = 30
        self.margin = 30
        self.builder = GUIBuilder(self.root, self)

    def start(self):
        self.builder.build()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.destroy()

    def handle_click(self, event):
        if not self.engine.board or self.engine.is_game_over:
            return
        c = round((event.x - self.margin) / self.cell_size)
        r = round((event.y - self.margin) / self.cell_size)
        try:
            self.engine.make_move(r, c)
        except CustomError as e:
            messagebox.showerror("错误", str(e))

    def update_view(self, model, message=None):
        if message: 
            clean_msg = str(message).replace("GAME_OVER:", "")
            if "GAME_OVER" in str(message):
                messagebox.showinfo("游戏结束", clean_msg)
            self.builder.update_status(clean_msg)
            
        if model.board:
            self.draw_board(model.board)
            p_name = "黑方" if model.current_player == PieceType.BLACK else "白方"
            status = "游戏结束" if model.is_game_over else f"当前执子: {p_name}"
            self.builder.update_info(f"{status} | {'五子棋' if model.game_type == GameType.GOMOKU else '围棋'}")

    def draw_board(self, board):
        canvas = self.builder.canvas
        canvas.delete("all")
        sz = board.size
        
        # 背景
        canvas.create_rectangle(0, 0, sz*self.cell_size + self.margin*2, sz*self.cell_size + self.margin*2, fill="#E3CF57")

        for i in range(sz):
            start = self.margin + i * self.cell_size
            end = self.margin + (sz-1)*self.cell_size
            canvas.create_line(self.margin, start, end, start)
            canvas.create_line(start, self.margin, start, end)

        r_offset = self.cell_size // 2 - 2
        for r in range(sz):
            for c in range(sz):
                piece = board.get_piece(r, c)
                if piece != PieceType.EMPTY:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    color = "black" if piece == PieceType.BLACK else "white"
                    canvas.create_oval(x-r_offset, y-r_offset, x+r_offset, y+r_offset, fill=color)

class GUIBuilder:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.canvas = None
        self.status_label = None
        self.info_label = None

    def build(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Button(control_frame, text="五子棋(15)", command=lambda: self.start_game(GameType.GOMOKU, 15)).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="围棋(19)", command=lambda: self.start_game(GameType.GO, 19)).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="悔棋", command=self.do_undo).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="虚着", command=self.do_pass).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="保存", command=self.do_save).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="读取", command=self.do_load).pack(side=tk.LEFT, padx=2)

        self.info_label = tk.Label(self.root, text="请开始游戏", font=("SimHei", 12, "bold"))
        self.info_label.pack(pady=5)

        self.canvas = tk.Canvas(self.root, width=500, height=500, bg="#E3CF57")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.controller.handle_click)

        self.status_label = tk.Label(self.root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def start_game(self, g_type, size):
        try:
            self.controller.engine.start_game(g_type, size)
            dim = size * 30 + 60
            self.canvas.config(width=dim, height=dim)
        except Exception as e: messagebox.showerror("错误", str(e))

    def do_undo(self): 
        try: self.controller.engine.undo()
        except Exception as e: messagebox.showwarning("提示", str(e))
    
    def do_pass(self):
        try: self.controller.engine.pass_turn()
        except Exception as e: messagebox.showwarning("提示", str(e))

    def do_save(self):
        fname = filedialog.asksaveasfilename()
        if fname: 
            try: self.controller.engine.save_game(fname)
            except Exception as e: messagebox.showerror("错误", str(e))

    def do_load(self):
        fname = filedialog.askopenfilename()
        if fname: 
            try: self.controller.engine.load_game(fname)
            except Exception as e: messagebox.showerror("错误", str(e))

    def update_status(self, text): self.status_label.config(text=text)
    def update_info(self, text): self.info_label.config(text=text)

# ==========================================
# 6. 主程序入口
# ==========================================

if __name__ == "__main__":
    game_engine = GameEngine()
    
    print("\n=== 通用棋类对战平台 ===")
    print("检测到您正在运行的环境可能不支持GUI...")
    print("选择启动模式: 1. 控制台 (推荐)  2. 图形界面GUI")
    
    mode = input(">> ").strip()
    
    if mode == '2':
        try:
            app = GraphicalUI(game_engine)
            app.start()
        except (RuntimeError, ImportError, tk.TclError) as e:
            print(f"\n[错误] 启动图形界面失败: {e}")
            print(">>> 检测到无显示环境 (Headless Environment)，自动切换至控制台模式。")
            app = ConsoleUI(game_engine)
            app.start()
        except Exception as e:
            print(f"发生未知错误: {e}")
    else:
        app = ConsoleUI(game_engine)
        app.start()