import json
from hydrodiy.io.hyruns import SiteBatch

SITEIDS_DEBUG =[
        405218, \
        234201, \
        405240, \
        401013, \
        410038, \
        219017
]


def select_sites(sites, debug, nbatch, taskid):
    """ Select sites depending on debug and batch modes """
    # Restricted list of sites
    if debug:
        return sites.loc[SITEIDS_DEBUG]

    # Define batch of sites
    if taskid>=0 and not debug:
        assert nbatch>1
        sb = SiteBatch(sites.index, nbatch)
        siteids = sb[taskid]
        return sites.loc[siteids]

    return sites
