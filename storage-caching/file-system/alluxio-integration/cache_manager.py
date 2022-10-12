#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Futurewei Technologies.
# Copyright: All rights reserved (2022+)
# Author: Nikunj J Parekh
# Email: nparekh@futurewei.com

from abc import ABCMeta
from abc import abstractmethod

# The Abstract Base Class that will handle all Dataset Cache Management
class CacheManagerABC(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def abort(self):
        pass


# The Abstract Base Class that will handle all Dataset Cache Management
class ExecutorServiceABC(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def _getSpaceNeededForDatasetInBytes(self):
        pass

    @abstractmethod
    def _getCapacityBytes(self):
        pass

    @abstractmethod
    def _getUsedBytes(self):
        pass
