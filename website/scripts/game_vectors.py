"""Interpretable target-relative, phase-aware fingerprints. No learned weights.

Observe the target player's positions after their moves, starting after 12
opening plies. Pool three equal chronological segments. Block scaling avoids
letting the 48 piece-square features drown out pawn structure or king safety.
"""
import re
import chess
import numpy as np

PIECES = list(chess.PIECE_TYPES)
START = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
BLOCK_WEIGHTS = {'placement': 1.0, 'pawns': 1.5, 'activity': 1.25, 'king': 1.0, 'material': .5, 'decisions': 1.0}


def normalize(value):
    return re.sub('[^a-z]', '', value.lower())


def target_color(game, player):
    key = normalize(player)
    for field, color in [('White', chess.WHITE), ('Black', chess.BLACK)]:
        name = game.headers.get(field, '')
        tokens = re.findall('[a-z]+', name.lower())
        if key in tokens or key == normalize(name):
            return color
    raise ValueError(f'Cannot identify {player} in game headers')


def position_features(board, color):
    values, names, groups = [], [], []
    def add(group, name, value):
        groups.append(group); names.append(name); values.append(float(value))
    def relative_rank(square):
        rank = chess.square_rank(square)
        return rank if color else 7-rank
    own = board.occupied_co[color]
    enemy = board.occupied_co[not color]
    for piece in PIECES:
        squares = board.pieces(piece, color)
        # Two file halves, four ranks bands; orientation follows the target side.
        for rank_band in range(4):
            for file_half in range(2):
                count = sum(relative_rank(s)//2 == rank_band and chess.square_file(s)//4 == file_half for s in squares)
                add('placement', f'{chess.piece_name(piece)} placement rank{rank_band} file{file_half}', count/START[piece])
    pawns = board.pieces(chess.PAWN, color)
    opponents = board.pieces(chess.PAWN, not color)
    files = [sum(chess.square_file(s)==f for s in pawns) for f in range(8)]
    for f, count in enumerate(files): add('pawns', f'pawns on {chess.FILE_NAMES[f]} file', count/2)
    isolated = sum(not any(files[f2] for f2 in [f-1,f+1] if 0<=f2<8) for s in pawns for f in [chess.square_file(s)])
    passed = sum(not any(abs(chess.square_file(s)-chess.square_file(t))<=1 and relative_rank(t)>relative_rank(s) for t in opponents) for s in pawns)
    connected = sum(any(abs(chess.square_file(s)-chess.square_file(t))==1 and abs(chess.square_rank(s)-chess.square_rank(t))<=1 for t in pawns) for s in pawns)
    islands = sum(count>0 and (i==0 or files[i-1]==0) for i,count in enumerate(files))
    for name,value in [('doubled pawns',sum(max(0,n-1) for n in files)/8),('isolated pawns',isolated/8),('passed pawns',passed/8),('connected pawns',connected/8),('pawn islands',islands/4),('pawn advancement',sum(relative_rank(s) for s in pawns)/max(1,len(pawns))/7),('central pawns',sum(chess.square_file(s) in [3,4] for s in pawns)/4),('pawn count',len(pawns)/8)]: add('pawns',name,value)
    attacked = set()
    for square in chess.scan_forward(own): attacked.update(board.attacks(square))
    add('activity','center control',sum(s in attacked for s in [chess.D4,chess.E4,chess.D5,chess.E5])/4)
    add('activity','opponent half control',sum(relative_rank(s)>=4 for s in attacked)/32)
    add('activity','board control',len(attacked)/64)
    add('activity','undefended pieces',sum(not board.is_attacked_by(color,s) for s in chess.scan_forward(own) if board.piece_type_at(s)!=chess.KING)/15)
    mobility_board = board.copy(stack=False)
    if mobility_board.turn != color:
        mobility_board.turn = color
        mobility_board.ep_square = None
    legal = list(mobility_board.legal_moves)
    for piece in PIECES:
        add('activity',f'{chess.piece_name(piece)} mobility',sum(board.piece_type_at(m.from_square)==piece for m in legal)/32)
    add('activity','attacked enemy pieces',sum(s in attacked for s in chess.scan_forward(enemy))/16)
    add('activity','bishop pair',len(board.pieces(chess.BISHOP,color))>=2)
    king = board.king(color)
    if king is None: raise ValueError('Missing king')
    zone = set(board.attacks(king)) | {king}
    shield = sum(chess.square_distance(king,s)<=1 and relative_rank(s)>relative_rank(king) for s in pawns)
    for name,value in [('king advancement',relative_rank(king)/7),('king centrality',1-abs(chess.square_file(king)-3.5)/3.5),('pawn shield',shield/3),('king zone attacked',sum(board.is_attacked_by(not color,s) for s in zone)/len(zone)),('king zone defended',sum(board.is_attacked_by(color,s) for s in zone)/len(zone)),('king file open',files[chess.square_file(king)]==0),('castling rights',board.has_castling_rights(color))]: add('king',name,value)
    for piece in PIECES[:-1]:
        add('material',f'{chess.piece_name(piece)} balance',(len(board.pieces(piece,color))-len(board.pieces(piece,not color)))/START[piece])
    add('material','remaining material',sum(len(board.pieces(p,c))*v for p,v in [(1,1),(2,3),(3,3),(4,5),(5,9)] for c in chess.COLORS)/78)
    return values,names,groups


def strategic_features(game, player):
    color=target_color(game,player)
    board=game.board(); observations=[]
    for index,move in enumerate(game.mainline_moves()):
        mover=board.turn; piece=board.piece_type_at(move.from_square)
        decision=[board.is_capture(move),board.gives_check(move),piece==chess.PAWN,board.is_castling(move),chess.square_file(move.to_square) in [3,4],(chess.square_rank(move.to_square)-chess.square_rank(move.from_square))*(1 if color else -1)/7,chess.square_distance(move.from_square,move.to_square)/7]
        board.push(move)
        if index>=12 and mover==color:
            values,names,groups=position_features(board,color)
            decision_names=['capture rate','check rate','pawn move rate','castling rate','central file destination','forward movement','move distance']
            observations.append(values+decision)
        # Cap at 120 plies so marathon games do not dominate the representation.
        if index>=119: break
    if len(observations)<3: raise ValueError('Too few target decisions after opening')
    names += decision_names; groups += ['decisions']*len(decision_names)
    # Three chronological segments, not claims of opening/middlegame/endgame.
    chunks=np.array_split(np.asarray(observations,dtype=float),3)
    features=np.concatenate([chunk.mean(axis=0) for chunk in chunks])
    feature_names=[f'{phase}: {name}' for phase in ['early','middle','late'] for name in names]
    block_names=groups*3
    return features,feature_names,block_names
