// This file is AUTOGENERATED, do not edit.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "python/definitions.h"

namespace {
PyObject *TLczeroExceptionExceptionType;
struct TGameStateClassType;
extern PyTypeObject objGameStateClassType;
struct TGameStateClassType {
  PyObject_HEAD
  lczero::python::GameState *value;
};

PyObject* FGameStateMethodmoves(TGameStateClassType* self, PyObject* /* not used */) {
  PyObject *retval;
  std::vector<std::string> retval_cpp = self->value->moves();
  retval = PyList_New(retval_cpp.size());
  for (size_t i = 0; i < retval_cpp.size(); ++i) {
    const std::string& s = retval_cpp[i];
    PyList_SetItem(retval, i, Py_BuildValue("s#", s.data(), s.size()));
  }
  return retval;
}

PyObject* FGameStateMethodpolicy_indices(TGameStateClassType* self, PyObject* /* not used */) {
  PyObject *retval;
  std::vector<int> retval_cpp = self->value->policy_indices();
  retval = PyTuple_New(retval_cpp.size());
  for (size_t i = 0; i < retval_cpp.size(); ++i) {
    PyTuple_SetItem(retval, i, Py_BuildValue("i", retval_cpp[i]));
  }
  return retval;
}

PyObject* FGameStateMethodas_string(TGameStateClassType* self, PyObject* /* not used */) {
  PyObject *retval;
  const std::string& retval_cpp = self->value->as_string();
  retval = Py_BuildValue("s#", retval_cpp.data(), retval_cpp.size());
  return retval;
}

PyMethodDef rgGameStateClassFunctions[] = {
  {"moves", reinterpret_cast<PyCFunction>(&FGameStateMethodmoves), METH_NOARGS, nullptr},
  {"policy_indices", reinterpret_cast<PyCFunction>(&FGameStateMethodpolicy_indices), METH_NOARGS, nullptr},
  {"as_string", reinterpret_cast<PyCFunction>(&FGameStateMethodas_string), METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

int FGameStateConstructor(TGameStateClassType* self, PyObject *args, PyObject *kwargs) {
  const char* fen = nullptr;
  Py_ssize_t fen_len = 0;
  PyObject* moves = nullptr;
  const char* keywords[] = {"fen", "moves", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args,
      kwargs,
      "|z#O!",
      const_cast<char**>(keywords),
      &fen,
      &fen_len,
      &PyList_Type,
      &moves)) {
    return -1;
  }
  std::optional<std::string> fen_cpp;
  if (fen != nullptr) fen_cpp.emplace(fen, fen_len);
  std::vector<std::string> moves_cpp;
  if (moves != nullptr) {
    moves_cpp.reserve(PyList_Size(moves));
    for (Py_ssize_t i = 0; i < PyList_Size(moves); ++i) {
      PyObject* tmp = PyList_GetItem(moves, i);
      if (!PyUnicode_Check(tmp)) {
        PyErr_SetString(PyExc_TypeError, "String type expected.");
        return -1;
      }
      Py_ssize_t size;
      const char* str = PyUnicode_AsUTF8AndSize(tmp, &size);
      moves_cpp.emplace_back(str, size);
    }
  }
  try {
    self->value = new lczero::python::GameState(fen_cpp, moves_cpp);
  } catch (const lczero::Exception &ex) {
    PyErr_SetString(TLczeroExceptionExceptionType, ex.what());
    return -1;
  }
  return 0;
}

void FGameStateDestructor(TGameStateClassType* self) {
  delete self->value;
  Py_TYPE(self)->tp_free(&self->ob_base);
}

PyTypeObject objGameStateClassType = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "libprobs_chess.GameState",
  .tp_basicsize = sizeof(TGameStateClassType),
  .tp_dealloc = reinterpret_cast<destructor>(FGameStateDestructor),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = nullptr,
  .tp_methods = rgGameStateClassFunctions,
  .tp_init = reinterpret_cast<initproc>(FGameStateConstructor),
  .tp_alloc = PyType_GenericAlloc,
  .tp_new = PyType_GenericNew,
};
PyMethodDef rglibprobs_chessModuleFunctions[] = {
  {nullptr, nullptr, 0, nullptr}
};

PyModuleDef Tlibprobs_chessModule = {
  PyModuleDef_HEAD_INIT,
  "libprobs_chess",
  nullptr,
  -1,
  rglibprobs_chessModuleFunctions,
  nullptr, nullptr, nullptr, nullptr, 
};
}  // anonymous namespace

PyMODINIT_FUNC PyInit_libprobs_chess() {
  lczero::InitializeMagicBitboards();

  PyObject* module = PyModule_Create(&Tlibprobs_chessModule);
  if (module == nullptr) return nullptr;
  TLczeroExceptionExceptionType = PyErr_NewException("libprobs_chess.LczeroException", nullptr, nullptr);
  if (TLczeroExceptionExceptionType == nullptr) return nullptr;
  Py_INCREF(TLczeroExceptionExceptionType);
  PyModule_AddObject(module, "LczeroException", TLczeroExceptionExceptionType);
  if (PyType_Ready(&objGameStateClassType) != 0) return nullptr;
  PyModule_AddObject(module, "GameState", &objGameStateClassType.ob_base.ob_base);
  return module;
}
