#ifndef __GDS_UNIT_TEST_H__
#define __GDS_UNIT_TEST_H__

#include <Python.h>

PyObject * add(PyObject *, PyObject *);
PyObject * gds_system(PyObject *self, PyObject *args);
PyObject * std_standard_dev(PyObject *self, PyObject* args);
PyObject * test_read_image(PyObject* self, PyObject* args);
PyObject* test_read_image_data(PyObject* self, PyObject* args);
#endif