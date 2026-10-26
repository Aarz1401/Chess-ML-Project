"""Validate exported data against legal PGN replay and the feature projection."""
import io
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from build_atlas import describe
from game_vectors import strategic_features, position_features, target_color, BLOCK_WEIGHTS

# Substring matching would mistake Praggnanandhaa for Anand.
fixture=chess.pgn.Game()
fixture.headers['White']='Praggnanandhaa,R'
fixture.headers['Black']='Anand,V'
assert target_color(fixture,'Anand') == chess.BLACK
assert target_color(fixture,'Praggnanandhaa') == chess.WHITE

path = Path(__file__).resolve().parents[1] / 'dist' / 'atlas.json'
data = json.loads(path.read_text())
assert data['schemaVersion'] == 2
assert len({g['id'] for g in data['games']}) == len(data['games'])
for g in data['games']:
    game = chess.pgn.read_game(io.StringIO(g['pgn']))
    assert game and not game.errors
    rebuilt = describe(game, g['player'])
    assert rebuilt['id'] == g['id']
    assert rebuilt['moves'] == g['moves']
    assert rebuilt['fens'] == g['fens']
    np.testing.assert_allclose(rebuilt['features'], g['features'], atol=1e-6)
    strategic, names, blocks = strategic_features(game, g['player'])
    np.testing.assert_allclose(strategic, g['strategicFeatures'], atol=1e-7)
    assert len(g['vector']) == data['representation']['dimensions'] == len(names) == 288
    assert np.isfinite(g['vector']).all() and np.isfinite(g['position']).all()
raw = np.array([g['features'] for g in data['games']])
vectors = np.array([g['baselineVector'] for g in data['games']])
np.testing.assert_allclose(StandardScaler().fit_transform(raw), vectors, atol=2e-5)
projected = PCA(n_components=2, svd_solver='full').fit_transform(vectors)
np.testing.assert_allclose(projected, [g['baselinePosition'] for g in data['games']], atol=2e-5)
strategic_raw=np.array([g['strategicFeatures'] for g in data['games']])
scaled=np.clip(StandardScaler().fit_transform(strategic_raw),-3,3)
blocks=data['representation']['blocks']
weights=np.array([np.sqrt(BLOCK_WEIGHTS[group]/blocks.count(group)) for group in blocks])
expected=scaled*weights
expected/=np.maximum(np.linalg.norm(expected,axis=1,keepdims=True),1e-12)
np.testing.assert_allclose(expected,[g['vector'] for g in data['games']],atol=2e-6)
# Color/rank mirroring must preserve all target-relative position features.
for g in data['games'][::20]:
    board=chess.Board(g['fens'][min(20,len(g['fens'])-1)])
    for color in chess.COLORS:
        a,_,_=position_features(board,color)
        b,_,_=position_features(board.mirror(),not color)
        np.testing.assert_allclose(a,b,atol=1e-10)
print(f"PASS: {len(data['games'])} unique games, legal replay, feature extraction, standardization, PCA, 288D strategic fingerprints, color symmetry")
