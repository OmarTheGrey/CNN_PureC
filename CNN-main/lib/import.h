/*
 * import.h — MNIST loader prototypes
 * ----------------------------------
 * Declares helpers for reading images and labels from IDX files.
 */

#ifndef IMPORT_H
#define IMPORT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

int* readParameters(char* filename);
double*** readImages(char* filename);
int* readLabels(char* filename);

#endif