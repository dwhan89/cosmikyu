os.environ["PYSM_LOCAL_DATA"] = "/"
DEFAULT_TCMB = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e10


class mmDLHelper(object):
    def __init__(self, path):
        self.path = path
        self.freqs = [30, 90, 148, 219, 277, 350]
        self.compts = ["lensed_cmb", "kappa", "ksz", "tsz", "rad_pts", "ir_pts"]
        self.input_units = u.uK_CMB

    def get_path(self, sim_idx):
        return self.path

    def get_file_name(self, compt_idx, sim_idx, freq=None, polidx=None):
        # rpath = self.get_path(sim_idx)
        if compt_idx in ["tsz", "rad_pts", "ir_pts"]:
            assert (freq in self.freqs)
            f = os.path.join(self.path, f"{sim_idx:05d}/{compt_idx}_{freq:03d}ghz_{sim_idx:05d}.fits")
        elif compt_idx in ["kappa", "ksz"]:
            f = os.path.join(self.path, f"{sim_idx:05d}/{compt_idx}_{sim_idx:05d}.fits")
        elif compt_idx in ["lensed_cmb"]:
            assert (polidx in ["T", "Q", "U"])
            f = os.path.join(self.path, f"{sim_idx:05d}/{compt_idx}_{polidx}_{sim_idx:05d}.fits")
        elif compt_idx in ["combined"]:
            assert (freq in self.freqs)
            assert (polidx in ["T", "Q", "U"])
            f = os.path.join(self.path, f"{sim_idx:05d}/{compt_idx}_{polidx}_{freq:03d}ghz_{sim_idx:05d}.fits")
        else:
            raise NotImplemented()
        return f

    def fnu(self, nu, tcmb=DEFAULT_TCMB):
        """
        nu in GHz
        tcmb in Kelvin
        """
        nu = np.asarray(nu)
        mu = H_CGS * (1e9 * nu) / (K_CGS * tcmb)
        ans = mu / np.tanh(old_div(mu, 2.0)) - 4.0
        return ans


class mmDLModel(Model):
    def __init__(
            self,
            path,
            map_dist=None,
            verbose=False,
    ):
        self.path = path  ## root directory for mmdl
        self.map_dist = map_dist
        self.verbose = verbose
        self.mmdl_helper = mmDLHelper(self.path)

        ## fixed for now
        self.nside = 4096  ## HEALPix NSIDE of the output maps
        self.FREQ_INDP = -1
        self.interpolation_kind = "linear"

        self.freq_independent_compts = ["lensed_cmb_t", "lensed_cmb_q", "lensed_cmb_u", "kappa", "ksz"]
        self.freq_dependent_compts = ["tsz", "rad_pts", "ir_pts"]

        self.interpolators = {}

    def clean_cached(self):
        for item in self.interpolators:
            item.cached_maps.clear()
        del self.interpolators
        self.interpolators = {}

    def get_interpolators(self, sim_idx):
        if sim_idx not in self.interpolators:
            self.interpolators[sim_idx] = {}

            for compt_idx in ["lensed_cmb_t", "lensed_cmb_q", "lensed_cmb_u", "kappa", "ksz"]:
                self.interpolators[sim_idx][compt_idx] = mmDLInterpolatingComponentIdentity(nside=self.nside,
                                                                                            sim_idx=sim_idx,
                                                                                            compt_idx=compt_idx,
                                                                                            mmdl_helper=self.mmdl_helper,
                                                                                            interpolation_kind=self.interpolation_kind,
                                                                                            map_dist=self.map_dist,
                                                                                            verbose=self.verbose)

            ## use analytic expression for ksz
            self.interpolators[sim_idx]["tsz"] = mmDLInterpolatingComponentTSZ(nside=self.nside,
                                                                               sim_idx=sim_idx,
                                                                               compt_idx=compt_idx,
                                                                               mmdl_helper=self.mmdl_helper,
                                                                               interpolation_kind=self.interpolation_kind,
                                                                               map_dist=self.map_dist,
                                                                               verbose=self.verbose)

            for compt_idx in ["rad_pts", "ir_pts"]:
                self.interpolators[sim_idx][compt_idx] = mmDLInterpolatingComponentPTSource(nside=self.nside,
                                                                                            sim_idx=sim_idx,
                                                                                            compt_idx=compt_idx,
                                                                                            mmdl_helper=self.mmdl_helper,
                                                                                            interpolation_kind=self.interpolation_kind,
                                                                                            map_dist=self.map_dist,
                                                                                            verbose=self.verbose)
        return self.interpolators[sim_idx]

    def get_emission(self, sim_idx, freqs: u.GHz, weights=None):
        ret = {}
        for compt_idx in self.freq_independent_compts + self.freq_dependent_compts:
            interpolators = self.get_interpolators(sim_idx)
            ret[compt_idx] = interpolators[compt_idx].get_emission(freqs, weights=None)
        ret["combined_t"] = ret["lensed_cmb_t"] + ret["ksz"] + ret["tsz"] + ret["rad_pts"] + ret["ir_pts"]
        return ret

    '''
    def read_map(self, path, unit=None, field=0, nside=None):
        warnings.warn(f"[WARNING] read_map is override by {self.__class__.__name__}")
        output_map = hp.read_map(path, field=field, verbose=False, dtype=None)
        dtype = np.dtype(np.float32)
        nside_in = 8192
        nside = nside_out = self.nside
        # numba only supports little endian
        if nside < nside_in:  # do downgrading in double precision
            output_map = hp.ud_grade(output_map.astype(np.float64), nside_out)
        else:
            output_map = hp.ud_grade(output_map, nside_out=nside)
        output_map = output_map.astype(dtype, copy=False)

        return u.Quantity(output_map, unit, copy=False)
    '''


class mmDLInterpolatingComponent(InterpolatingComponent):
    def __init__(
            self,
            nside,
            sim_idx,
            compt_idx,
            mmdl_helper,
            interpolation_kind="linear",
            map_dist=None,
            verbose=False,
    ):
        self.sim_idx = sim_idx
        if "cmb" in compt_idx:
            tokens = compt_idx.split("_")
            self.compt_idx = "_".join(tokens[:2])
            self.polidx = tokens[-1].upper()
        else:
            self.compt_idx = compt_idx
            self.polidx = None

        self.mmdl_helper = mmdl_helper

        super().__init__(path=mmdl_helper.get_path(sim_idx),
                         input_units=mmdl_helper.input_units,
                         nside=nside,
                         interpolation_kind=interpolation_kind,
                         map_dist=map_dist,
                         verbose=verbose)

    def get_filenames(self, path):
        # Override this to implement name convention
        filenames = {}
        for freq in self.mmdl_helper.freqs:
            filenames[freq] = self.mmdl_helper.get_file_name(self.compt_idx, self.sim_idx, freq=freq,
                                                             polidx=self.polidx)
        return filenames

    def read_map(self, path, unit=None, field=0, nside=None):
        warnings.warn(f"[WARNING] read_map is override by mmDLInterpolatingComponent")
        output_map = hp.read_map(path, field=field, verbose=False, dtype=None)
        dtype = np.dtype(np.float32)
        nside_in = 8192
        nside = nside_out = self.nside
        # numba only supports little endian
        if nside < nside_in:  # do downgrading in double precision
            output_map = hp.ud_grade(output_map.astype(np.float64), nside_out)
        else:
            output_map = hp.ud_grade(output_map, nside_out=nside)
        output_map = output_map.astype(dtype, copy=False)

        return u.Quantity(output_map, unit, copy=False)


class mmDLInterpolatingComponentIdentity(mmDLInterpolatingComponent):
    '''
        Mock Interpolater for frequency independent component -> always return the same saved map
    '''

    def __init__(
            self,
            nside,
            sim_idx,
            compt_idx,
            mmdl_helper,
            interpolation_kind="linear",
            map_dist=None,
            verbose=False,
    ):

        super().__init__(
            nside=nside,
            sim_idx=sim_idx,
            compt_idx=compt_idx,
            mmdl_helper=mmdl_helper,
            interpolation_kind=interpolation_kind,
            map_dist=map_dist,
            verbose=verbose)

    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        nu = utils.check_freq_input(freqs)
        m = self.read_map_by_frequency(148) * u.uK_RJ

        ### Note that kappa is dimensionless. Remove the dimension
        if self.compt_idx == "kappa":
            m = m.to(self.mmdl_helper.input_units, equivalencies=u.cmb_equivalencies(148 * u.GHz)).value

        out = {}
        for nu_target in nu:
            out[nu_target] = m.copy()
        return out


class mmDLInterpolatingComponentTSZ(mmDLInterpolatingComponent):
    '''
        Mock Interpolater for frequency independent component -> always return the same saved map
    '''

    def __init__(
            self,
            nside,
            sim_idx,
            compt_idx,
            mmdl_helper,
            interpolation_kind="linear",
            map_dist=None,
            verbose=False,
    ):
        super().__init__(
            nside=nside,
            sim_idx=sim_idx,
            compt_idx=compt_idx,
            mmdl_helper=mmdl_helper,
            interpolation_kind=interpolation_kind,
            map_dist=map_dist,
            verbose=verbose)

    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        nu = utils.check_freq_input(freqs)
        m = self.read_map_by_frequency(148) * u.uK_RJ
        m = m.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(148 * u.GHz))

        out = {}
        for nu_target in nu:
            out[nu_target] = m * self.mmdl_helper.fnu(nu_target) / self.mmdl_helper.fnu(148)
            out[nu_target] = out[nu_target].to(u.uK_RJ, equivalencies=u.cmb_equivalencies(nu_target * u.GHz))
        return out


class mmDLInterpolatingComponentPTSource(mmDLInterpolatingComponent):
    '''
        Mock Interpolater for frequency independent component -> always return the same saved map
    '''

    def __init__(
            self,
            nside,
            sim_idx,
            compt_idx,
            mmdl_helper,
            interpolation_kind="linear",
            map_dist=None,
            verbose=False,
    ):

        super().__init__(
            nside=nside,
            sim_idx=sim_idx,
            compt_idx=compt_idx,
            mmdl_helper=mmdl_helper,
            interpolation_kind=interpolation_kind,
            map_dist=map_dist,
            verbose=verbose)

    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:

        nu = utils.check_freq_input(freqs)
        out = {}
        if len(nu) == 1:

            # special case: we request only 1 frequency and that is among the ones
            # available as input
            check_isclose = np.isclose(self.freqs, nu[0])
            if np.any(check_isclose):
                freq = self.freqs[check_isclose][0]
                out[freq] = self.read_map_by_frequency(freq) * u.uK_RJ
                return out

        npix = hp.nside2npix(self.nside)
        if nu[0] < self.freqs[0]:
            warnings.warn(
                "Frequency not supported, requested {} Ghz < lower bound {} GHz".format(
                    nu[0], self.freqs[0]
                )
            )
            return np.zeros((3, npix)) << u.uK_RJ
        if nu[-1] > self.freqs[-1]:
            warnings.warn(
                "Frequency not supported, requested {} Ghz > upper bound {} GHz".format(
                    nu[-1], self.freqs[-1]
                )
            )
            return np.zeros((3, npix)) << u.uK_RJ

        for nu_target in nu:
            weights = utils.normalize_weights([nu_target], weights)
            ## we use the closest two bandpath to interpolate
            first_freq_i, last_freq_i = np.searchsorted(self.freqs, [freq])
            first_freq_i -= 1
            last_freq_i += 1

            for freq in freq_range:
                if freq not in self.cached_maps:
                    m = self.read_map_by_frequency(freq) * u.uK_RJ
                    ## put in the intensity unit
                    m = m.to(u.Jy, equivalencies=u.cmb_equivalencies(freq * u.GHz)).value
                    if m.shape[0] != 3:
                        m = m.reshape((1, -1))
                    self.cached_maps[freq] = m.astype(np.float32)
                    if self.verbose:
                        for i_pol, pol in enumerate("IQU" if m.shape[0] == 3 else "I"):
                            print(
                                "Mean emission at {} GHz in {}: {:.4g} uK_RJ".format(
                                    freq, pol, self.cached_maps[freq][i_pol].mean()
                                )
                            )
            mv1 = self.cached_maps[freq_range[0]]
            mv2 = self.cached_maps[freq_range[1]]

            m_out = np.zeros(mv1.shape)
            ## true source
            loc = np.where((mv1 > 0) & (mv2 > 0))
            m_out[loc] = compute_interpolated_emission_numba([nu_target],
                                                             weights,
                                                             freq_range,
                                                             {freq_range[0]: np.log(mv1[loc]),
                                                              freq_range[1]: np.log(mv2[loc])})
            ## ringing noise
            loc = np.where((mv1 <= 0) | (mv2 <= 0))
            m_out[loc] = compute_interpolated_emission_numba([nu_target],
                                                             weights,
                                                             freq_range,
                                                             {freq_range[0]: mv1[loc], freq_range[1]: mv2[loc]})
            m_out = m_out * u.Jy
            out[nu_target] = m_out.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(freq * u.GHz))

        return out
