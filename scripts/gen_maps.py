# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import random

import camb
import click

from lensit.clusterlens import lensingmap
from tqdm import tqdm
from pytorch_lightning import seed_everything

import sys

sys_path_folder = "../python/"
sys.path.append(sys_path_folder)
import flatsky, tools, lensing, misc


def get_cluster_maps(nsims, npix, lpix_amin, mass, z, **args):

    maps = []
    masses = []

    for M200 in mass:

        paramfile = "params.ini"
        print("\tread/get necessary params")
        param_dict = misc.get_param_dict(paramfile)

        data_folder = param_dict["data_folder"]

        # params or supply a params file
        dx = lpix_amin
        boxsize_amin = dx * npix
        nx = int(boxsize_amin / dx)
        mapparams = [nx, nx, dx]
        x1, x2 = -nx / 2.0 * dx, nx / 2.0 * dx
        verbose = 0
        pol = True  # param_dict['pol']
        debug = param_dict["debug"]
        # beam and noise levels
        noiseval = param_dict["noiseval"]  # uK-arcmin
        if pol:
            noiseval = [noiseval, noiseval * np.sqrt(2.0), noiseval * np.sqrt(2.0)]
        beamval = param_dict["beamval"]  # arcmins

        # CMB power spectrum
        cls_file = "%s/%s" % (param_dict["data_folder"], param_dict["cls_file"])

        if not pol:
            tqulen = 1
        else:
            tqulen = 3
        tqu_tit_arr = ["T", "Q", "U"]

        # sim stuffs
        total_sim_types = (
            2  # param_dict['total_sim_types'] #unlensed background and lensed clusters
        )
        total_clusters = nsims  # param_dict['total_clusters']
        total_randoms = (
            10 * nsims
        )  # param_dict['total_randoms'] #total_clusters * 10 #much more randoms to ensure we are not dominated by variance in background stack.

        # cluster info
        cluster_mass = M200 * 1e14  # param_dict['cluster_mass']
        cluster_z = z  # param_dict['cluster_z']

        # cluster mass definitions
        delta = param_dict["delta"]
        rho_def = param_dict["rho_def"]
        profile_name = param_dict["profile_name"]

        # cosmology
        h = param_dict["h"]
        omega_m = param_dict["omega_m"]
        omega_lambda = param_dict["omega_lambda"]
        z_lss = param_dict["z_lss"]
        T_cmb = param_dict["T_cmb"]

        # cutouts specs
        cutout_size_am = param_dict["cutout_size_am"]  # arcmins

        # for estimating cmb gradient
        apply_wiener_filter = param_dict["apply_wiener_filter"]
        lpf_gradient_filter = param_dict["lpf_gradient_filter"]
        cutout_size_am_for_grad = param_dict["cutout_size_am_for_grad"]  # arcminutes

        # get ra, dec or map-pixel grid
        ra = np.linspace(x1, x2, nx)  # arcmins
        dec = np.linspace(x1, x2, nx)  # arcmins
        ra_grid, dec_grid = np.meshgrid(ra, dec)

        # read Cls now
        el, dl_tt, dl_ee, dl_bb, dl_te = np.loadtxt(cls_file, unpack=1)
        dl_all = np.asarray([dl_tt, dl_ee, dl_bb, dl_te])
        cl_all = tools.dl_to_cl(el, dl_all)
        cl_tt, cl_ee, cl_bb, cl_te = cl_all  # Cls in uK
        cl_dic = {}
        cl_dic["TT"], cl_dic["EE"], cl_dic["BB"], cl_dic["TE"] = (
            cl_tt,
            cl_ee,
            cl_bb,
            cl_te,
        )
        if not pol:
            cl = [cl_tt]
        else:
            cl = cl_all
        # loglog(el, cl_tt)
        print(len(el))

        # get beam and noise
        bl = tools.get_bl(beamval, el, make_2d=1, mapparams=mapparams)
        nl_dic = {}
        if pol:
            nl = []
            for n in noiseval:
                nl.append(tools.get_nl(n, el))
            nl = np.asarray(nl)
            nl_dic["T"], nl_dic["P"] = nl[0], nl[1]
        else:
            nl = [tools.get_nl(noiseval, el)]
            nl_dic["T"] = nl[0]
        print("\tkeys in nl_dict = %s" % (str(nl_dic.keys())))

        # NFW lensing convergence
        ra_grid_deg, dec_grid_deg = ra_grid / 60.0, dec_grid / 60.0

        M200c_list = np.tile(cluster_mass, total_clusters)
        redshift_list = np.tile(cluster_z, total_clusters)
        ra_list = dec_list = np.zeros(total_clusters)

        kappa_arr = lensing.get_convergence(
            ra_grid_deg,
            dec_grid_deg,
            ra_list,
            dec_list,
            M200c_list,
            redshift_list,
            param_dict,
        )
        print("\tShape of convergence array is %s" % (str(kappa_arr.shape)))

        # perform CMB and noise simulations: unlensed background and lensed clusters
        sim_dic = {}
        for iter in range(total_sim_types):
            if iter == 0:  # cluster lensed sims
                do_lensing = True
                nsims_i = total_clusters
                sim_type = "clusters"
            else:
                do_lensing = False
                nsims_i = total_randoms
                sim_type = "randoms"
            sim_dic[sim_type] = {}
            print("\tcreating %s %s simulations" % (nsims_i, sim_type))
            sim_arr = []
            for i in tqdm(range(nsims_i)):
                if not pol:
                    cmb_map = np.asarray(
                        [flatsky.make_gaussian_realisation(mapparams, el, cl[0], bl=bl)]
                    )
                    noise_map = np.asarray(
                        [flatsky.make_gaussian_realisation(mapparams, el, nl[0])]
                    )
                else:
                    cmb_map = flatsky.make_gaussian_realisation(
                        mapparams,
                        el,
                        cl[0],
                        cl2=cl[1],
                        cl12=cl[3],
                        bl=bl,
                        qu_or_eb="qu",
                    )
                    noise_map_T = flatsky.make_gaussian_realisation(
                        mapparams, el, nl[0]
                    )
                    noise_map_Q = flatsky.make_gaussian_realisation(
                        mapparams, el, nl[1]
                    )
                    noise_map_U = flatsky.make_gaussian_realisation(
                        mapparams, el, nl[1]
                    )
                    noise_map = np.asarray([noise_map_T, noise_map_Q, noise_map_U])

                if do_lensing:
                    cmb_map_lensed = []
                    for tqu in range(tqulen):
                        unlensed_cmb = np.copy(cmb_map[tqu])
                        lensed_cmb = lensing.perform_lensing(
                            ra_grid_deg,
                            dec_grid_deg,
                            unlensed_cmb,
                            kappa_arr[i],
                            mapparams,
                        )
                        cmb_map_lensed.append(lensed_cmb)
                    cmb_map = np.asarray(cmb_map_lensed)

                sim_map = cmb_map + noise_map
                sim_arr.append(sim_map)
            sim_dic[sim_type]["sims"] = np.asarray(sim_arr)

        maps.append(sim_dic["clusters"]["sims"])
        masses.append([M200 for _ in range(nsims)])

    return (
        np.stack(maps, axis=0).reshape((nsims * len(mass), 3, npix, npix)),
        np.array(masses).flatten(),
    )


@click.command()
@click.argument("mass", nargs=-1, type=float)
@click.argument("destdir", nargs=1, type=click.Path(exists=False))
@click.option(
    "--nsims", help="Number of simulations for each mass", required=True, type=int
)
@click.option(
    "--npix",
    default=64,
    show_default=True,
    help="Number of pixels in each line/column of the map.",
)
@click.option(
    "--lpix_amin",
    default=0.3,
    show_default=True,
    help="Physical size of a pixel in arcmin",
)
@click.option("--z", default=0.7, show_default=True, help="Redshift")
@click.option("--seed", help="Seed used to generate the maps", default=None, type=int)
def genmaps(**args):
    """A program to generate CMB lensed maps. This also generates the unlensed and lensed CMB maps without noise."""
    if not args["seed"]:
        args["seed"] = random.randint(0, 2**32 - 1)
    if not os.path.exists(args["destdir"]):
        os.makedirs(args["destdir"])
    print("Seeding everything with seed {}...".format(args["seed"]))
    seed_everything(args["seed"])

    args["destdir"] = os.path.join(os.getcwd(), args["destdir"])
    os.environ["LENSIT"] = args["destdir"]
    if not os.path.exists(args["destdir"]):
        os.makedirs(args["destdir"])
    maps, masses = get_cluster_maps(**args)
    np.save(os.path.join(args["destdir"], "maps"), maps, allow_pickle=True)
    np.save(os.path.join(args["destdir"], "masses"), masses, allow_pickle=True)
    pickle.dump(args, open(os.path.join(args["destdir"], "args"), "wb"))


if __name__ == "__main__":
    genmaps()
