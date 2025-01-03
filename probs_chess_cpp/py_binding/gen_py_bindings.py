#!/usr/bin/env python3

#  This file is part of Leela Chess Zero.
#  Copyright (C) 2020 The LCZero Authors
#
#  Leela Chess is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Leela Chess is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
#
#  Additional permission under GNU GPL version 3 section 7
#
#  If you modify this Program, or any covered work, by linking or
#  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
#  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
#  modified version of those libraries), containing parts covered by the
#  terms of the respective license agreement, the licensors of this
#  Program grant you additional permission to convey the resulting work.

import sys
from pybind.writer import Writer
from pybind import Module, Class
from pybind.parameters import (StringParameter, ListOfStringsParameter, NumericParameter)
from pybind.retval import (StringRetVal, ListOfStringsRetVal, IntegralTupleRetVal)
from pybind.exceptions import CppException

# Module
mod = Module('libprobs_chess')
mod.AddInclude('python/definitions.h')
mod.AddInitialization('lczero::InitializeMagicBitboards();')
ex = mod.AddException(
    CppException('LczeroException', cpp_name='lczero::Exception'))

# ChessEnv class
game_state = mod.AddClass(
    Class('ChessEnv', cpp_name='probs::python::ChessEnv'))
game_state.constructor.AddParameter(
    StringParameter('fen', optional=True, can_be_none=True),
    NumericParameter('max_ply', optional=True)
).AddEx(ex)
game_state.AddMethod('legal_moves').AddRetVal(ListOfStringsRetVal())
game_state.AddMethod('move').AddParameter(StringParameter('move', optional=True, can_be_none=True))
game_state.AddMethod('game_state').AddRetVal(StringRetVal())
game_state.AddMethod('policy_indices').AddRetVal(IntegralTupleRetVal('i'))
game_state.AddMethod('as_string').AddRetVal(StringRetVal())

with open(sys.argv[1], 'wt') as f:
    writer = Writer(f)
    mod.Generate(writer)