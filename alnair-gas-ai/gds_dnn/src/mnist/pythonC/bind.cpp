#include "gds_unit_test.h"

// static PyMethodDef SpamMethods[] = {
//     {"system",  gds_system, METH_VARARGS,
//      "Execute a shell command."},
//     {"add",  add, METH_VARARGS,
//      "Execute a shell command."},
//     {NULL, NULL, 0, NULL}        /* Sentinel */
// };
// static struct PyModuleDef spammodule = {
//     PyModuleDef_HEAD_INIT,
//     "spam",   /* name of module */
//     NULL, /* module documentation, may be NULL */
//     -1,       /* size of per-interpreter state of the module,
//                  or -1 if the module keeps state in global variables. */
//     SpamMethods
// };

char system_docs[] = "call shell command by 'system'.  import gds_unittest as gds gds.system(\'ls -l\') ";
char addfunc_docs[] = "Add two numbers function.";
char stddevfunc_docs[] = "Return the standard deviation of a list.";
char gds_readimage_docs[] = "Read a batch of data from imagefile(mnist).";
char gds_readimagedata_docs[] = "Read a batch of data from imagefile(mnist). It returns the rows and columns of image";

static PyObject *initError;

static PyMethodDef unittest_funcs[] = {
	{	"system",
		(PyCFunction)gds_system,
		METH_VARARGS,
		system_docs},
	{	"add",
		(PyCFunction)add,
		METH_VARARGS,
		addfunc_docs},
	{	"standard_dev",
		(PyCFunction)std_standard_dev,
		METH_VARARGS,
		stddevfunc_docs},
	{	"gds_read_image",
		(PyCFunction)test_read_image,
		METH_VARARGS,
		gds_readimage_docs},
	{	"gds_read_image_data",
		(PyCFunction)test_read_image_data,
		METH_VARARGS,
		gds_readimage_docs},
	{	NULL}
};

char gds_unittest_docs[] = "This is a set of unit tests for gas-ai.";

static PyModuleDef unittests_mod = {
	PyModuleDef_HEAD_INIT,
	"unittests",
	gds_unittest_docs,
	-1,
	unittest_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit_unittests(void)
{
    PyObject *m;

    m = PyModule_Create(&unittests_mod);
    if (m == NULL)
        return NULL;

    initError = PyErr_NewException("GAS unit test.error", NULL, NULL);
    Py_XINCREF(initError);
    if (PyModule_AddObject(m, "error", initError) < 0) {
        Py_XDECREF(initError);
        Py_CLEAR(initError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}