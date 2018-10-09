"""
Query type related functions
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

import numpy as np


def check_query_type(type):
    """Check the query type.

    Only the following query types are allowed:
        AllowedType:
            AllLabels: Query all labels of an instance
            PartLabels: Query part of labels of an instance (Only available in multi-label setting)
        NotImplementedQueryType
            Relations: Query relations between two object
            Features: Query unlab_features of instances
            Examples: Query examples given constrains

    Parameters
    ----------
    type: str
        query type.

    Returns
    -------

    """
    assert (isinstance(type, str))
    QueryType = ['AllLabels', 'PartLabels']
    NotImplementedQueryType = ['Relations', 'Features', 'Examples']
    if type in QueryType:
        return True
    else:
        return False



'''
class QueryConfig():
    """Query Config.

    Query Config is the class that store/transform basic AL settings.
    Information stored here should be useful for recover from breakpoint,
    data split, validate query etc.

    Parameters
    ----------

    Examples
    --------
    """
    def __init__(self, query_type=None, datamat=None, label=None):
        if query_type is None:
            query_type = ['instance', 'all_labels']
        if query_type[0] not in ['instance', 'labels'] or query_type[1] not in ['instance', 'all_labels', 'part_labels', 'all_features', 'part_features']:
            raise NotImplemented(
                "%s is not Implemented" % (str(query_type)))

        if datamat is not None:
            Xsh = np.shape(datamat)
            if len(Xsh) == 0:
                raise TypeError("X must be list type.")
                # X = np.array(X)
            ysh = np.shape(label)
            label = np.array(label)
            assert (ysh[0] == Xsh[0])
            if len(ysh) == 2:
                self.classnum = ysh[1]
            else:
                self.classnum = ysh[0]
            self.featurenum = Xsh[1]
            self.instancenum = Xsh[0]
        self.query_type=query_type.copy()

    @property
    def query_type_info(self):
        return self.query_type

'''
