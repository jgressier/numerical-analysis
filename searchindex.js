Search.setIndex({"alltitles": {"2D FFT of space/time signal": [[14, null]], "Basics of discretization": [[11, null]], "Condition initiale et solution th\u00e9orique": [[4, "condition-initiale-et-solution-theorique"]], "Convection equation": [[11, null]], "Curve fitting": [[2, "curve-fitting"]], "Explicit Runge-Kutta methods": [[8, "explicit-runge-kutta-methods"]], "FFT of truncated signals": [[13, null]], "Finite Difference Method": [[0, null]], "Hydrogen-Oxygen Combustion Kinetics": [[10, "hydrogen-oxygen-combustion-kinetics"]], "Integration by the explicit euler method, upwind scheme": [[5, "integration-by-the-explicit-euler-method-upwind-scheme"]], "Integration of dynamic systems": [[11, null]], "Introduction": [[3, null], [4, null], [7, null], [15, null]], "Lax Wendroff scheme": [[5, "lax-wendroff-scheme"]], "POD of time signal": [[12, null]], "ROW and Gauss-Legendre implicit method": [[9, null]], "Runge-Kutta explicit methods": [[8, null]], "Signal processing": [[11, null]], "Spectral Collocation Methods": [[1, null]], "Spectral discretization": [[2, null]], "Stiffness": [[10, null]], "Upwind and Central scheme": [[5, null]], "Validation of Chebyshev derivation operator": [[2, "validation-of-chebyshev-derivation-operator"]], "Welcome to your Jupyter Book": [[11, null]], "von Neumann analysis": [[6, null]]}, "docnames": ["basics/Finite-Difference-Method", "basics/Spectral-Collocation-Methods", "basics/chebyshev-convergence", "basics/intro", "convection/intro", "convection/upwind", "convection/vneumann-analysis", "integration/intro", "integration/rk", "integration/row-gl", "integration/stiffness", "intro", "signal/POD1D", "signal/fft-hanning", "signal/fft2d", "signal/intro"], "envversion": {"sphinx": 62, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9}, "filenames": ["basics/Finite-Difference-Method.md", "basics/Spectral-Collocation-Methods.md", "basics/chebyshev-convergence.ipynb", "basics/intro.md", "convection/intro.ipynb", "convection/upwind.ipynb", "convection/vneumann-analysis.ipynb", "integration/intro.md", "integration/rk.ipynb", "integration/row-gl.ipynb", "integration/stiffness.ipynb", "intro.md", "signal/POD1D.ipynb", "signal/fft-hanning.ipynb", "signal/fft2d.ipynb", "signal/intro.md"], "indexentries": {}, "objects": {}, "objnames": {}, "objtypes": {}, "terms": {"": [4, 5, 10, 12], "0": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "01": 9, "02": 12, "04": [10, 14], "04089089e": 12, "06": 12, "07": 12, "08612071e": 12, "09152833e": 12, "0x7f617032c550": 4, "0x7f617034aa10": 4, "0x7f850cc69f90": 12, "0x7fe6041f7f50": 2, "0x7ff5cd074b10": 6, "1": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "10": [6, 9, 10, 12, 13, 14], "100": [4, 5, 6, 12, 13], "1000": [5, 10, 12], "101": 2, "11": 12, "119": 12, "12": [2, 10, 12], "120": 12, "121": 12, "125": 6, "1299": 12, "13": 2, "130": 14, "1300": 12, "1301": 12, "1302": 12, "1303": 12, "1304": 12, "1305": 12, "1306": 12, "13254048e": 12, "1383": 12, "1384": 12, "1385": 12, "1386": 12, "1387": 12, "14": [12, 14], "15": 2, "16": 13, "1966": 10, "1e": [6, 10], "1e1": 6, "1e4": 10, "1j": [6, 8], "1st": 5, "2": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "20": [9, 12, 14], "200": [4, 5, 12], "2048": 14, "21": 10, "21870389e": 12, "22158999e": 12, "23": 10, "24": 10, "25": [4, 5, 6, 9, 12], "26": 10, "26975264e": 12, "27": 10, "27279797e": 12, "2954509547": 12, "2b": 10, "2d": 11, "2k_1": 10, "2nd": 5, "2oh": 10, "2q": 5, "3": [4, 5, 6, 8, 9, 10, 12], "30": 14, "30j": 8, "31": 2, "3900": 14, "39084970e": 12, "3e": 12, "3e7": 10, "4": [2, 4, 5, 6, 8, 10, 12, 14], "40": 12, "40106483e": 12, "404": 12, "405": 12, "406": 12, "407": 12, "408": 12, "42697881e": 12, "43": 12, "47469002e": 12, "5": [2, 4, 5, 6, 8, 9, 10, 12, 13], "50": [2, 14], "500": 13, "512": 14, "53412180e": 12, "56902547e": 12, "5e": [5, 6], "6": [2, 8, 9, 10, 12, 14], "60j": 8, "611e": 12, "628e": 12, "7": 12, "70": 14, "72921764e": 12, "75": [4, 5], "79608020e": 12, "8": [2, 10, 13, 14], "81826533e": 12, "99915635e": 12, "999999999999916": 5, "A": [2, 8, 10, 12], "AT": 12, "At": 8, "But": 5, "If": 8, "In": [2, 10], "It": [2, 5, 11], "On": [4, 5], "One": 8, "The": [2, 5, 10], "These": 10, "To": [2, 12], "With": 5, "_": 9, "_2": 10, "__call__": 12, "__getitem__": 12, "__name__": 12, "_description_": 10, "_interv": 12, "_regist": 12, "_repr_html_": 12, "_type_": 10, "a11": 9, "a12": 9, "a21": 9, "a22": 9, "ab": [6, 8, 10, 12, 13, 14], "accept": 8, "accord": 7, "accur": 10, "aconv": [4, 5], "adt": 8, "advect": [4, 5], "after": 10, "aim": 15, "all": [5, 8], "alln": 2, "along": 10, "an": [2, 10, 12], "analysi": [11, 12], "analyz": 2, "ani": [10, 11, 12], "anim": [12, 14], "append": 9, "appropri": 12, "ar": [2, 5, 8, 10, 12], "arang": 2, "arbitrari": 10, "arg": 10, "argsort": 12, "argument": 8, "argwher": 12, "arrai": [5, 6, 8, 9], "assum": 2, "auto": 14, "avail": [2, 12], "avec": [4, 5], "averag": 12, "avoid": 12, "ax": [2, 6, 8, 12, 13, 14], "ax1": 2, "ax2": 2, "axef": [13, 14], "axek": 14, "axet": 13, "axi": [5, 6, 8], "axr": 6, "b": 10, "b1": 9, "b2": 9, "bad": 12, "base": [10, 15], "baseformatt": 12, "basi": 2, "bdf": 10, "been": 12, "being": 10, "between": 10, "bitrat": 12, "blatant": 5, "blit": 12, "both": 10, "boundari": 5, "build": 12, "bundl": 11, "butcher": [8, 9], "c": 10, "c1": 9, "c2": 9, "cach": 12, "cache_frame_data": 12, "calcul": [9, 10], "call": [10, 12], "can": [5, 8, 10, 12], "carefulli": 7, "cdot": [5, 8], "cell": [5, 10], "center": 5, "center2": 6, "center4": 6, "central": 11, "chang": 10, "characterist": 10, "characteristic_time_step": 10, "characteristic_time_step_robertson": 10, "chebcolloc": 2, "check": 11, "cherch": [4, 5], "choisit": [4, 5], "chosen": 7, "class": 8, "classic": [2, 5], "cmap": 8, "co": 12, "codec": 12, "coeffici": [8, 9], "colloc": [2, 11], "color": 8, "columnwis": 12, "combin": 5, "combustion_kinet": 10, "compar": 2, "complex": [6, 8], "comput": [2, 5, 10, 12], "concentr": 10, "condit": [5, 9], "consist": 8, "constant": 10, "content": 11, "continu": 2, "contour": 8, "contourf": 13, "contribut": 5, "convect": [4, 5], "converg": 2, "convert": 9, "copi": 5, "core": 12, "correl": 15, "coupl": [5, 10], "cov": 12, "covari": 12, "creat": 12, "crucial": 7, "cshift": 6, "current": [5, 10], "d": [2, 12, 13, 14], "da_dt": 10, "data": 12, "db_dt": 10, "dc_dt": 10, "de": [4, 5], "decomposit": [12, 15], "def": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "defin": [8, 9, 10], "definit": [4, 5], "delta": [5, 8], "depend": 5, "depth": 11, "der": 2, "deriv": 10, "describ": 10, "determin": 5, "df": 2, "df2": 2, "df_num": 2, "df_th": 2, "dfrac": 2, "diagram": 5, "did": 12, "differ": [5, 10, 11], "differenti": [2, 10], "dimens": 2, "disabl": 12, "discret": 5, "dispers": 6, "displai": 8, "dissip": [5, 6], "distribut": 5, "document": 11, "doe": 11, "domain": [4, 5], "dpi": 12, "dq": 8, "dt": [5, 8, 10], "du": [4, 5], "due": 10, "dx": 5, "dy1_dt": 10, "dy2_dt": 10, "dy3_dt": 10, "dy4_dt": 10, "dy5_dt": 10, "dy6_dt": 10, "dy_1": 10, "dy_2": 10, "dy_3": 10, "dy_4": 10, "dy_5": 10, "dy_6": 10, "e": 10, "each": [8, 10], "eigenvalu": [10, 12], "eigh": 12, "eigval": 10, "either": [5, 12], "elementari": 10, "elif": 12, "ell": [4, 5], "els": [10, 12], "embed_limit": 12, "endpoint": 2, "enumer": [2, 12, 13], "epsilon": 2, "equal": 8, "equat": [5, 10], "err": [2, 6], "error": [2, 5, 6], "et": 5, "evalu": [8, 10], "evolut": 12, "evolv": 10, "except": 5, "exp": [2, 4, 5, 6, 8, 12], "expans": 8, "expect": [2, 5], "explicit": [10, 11, 12], "extrapol": 2, "f": [2, 6, 9, 12, 14], "f_convpuls": 12, "f_oscpuls": 12, "f_per": 12, "f_pulse_p": 12, "f_steadypuls": 12, "f_xi": 2, "factori": 8, "fals": 12, "fast": 10, "faster": 10, "feel": 11, "few": [10, 11], "ffmpeg": 12, "fft": 11, "fft2": 14, "fftfreq": [13, 14], "fftshift": 14, "fig": [2, 6, 8, 12, 13, 14], "figsiz": [2, 6, 8, 9, 10, 12, 13, 14], "figur": [9, 10, 12], "file": [11, 12], "final": 5, "finit": 11, "first": [2, 5], "fit_to_gauss": 2, "fix": 9, "flip": 12, "flow": 5, "flux": 6, "fluxk": 6, "fluxmeth": 6, "fmt": 12, "forc": 5, "form": 5, "formatt": 12, "formul": 5, "four": 10, "fourier": 15, "fp": 12, "frac": [4, 5, 8, 10], "frac12": 5, "frame": [12, 14], "from": 10, "fromm": 6, "ft": 13, "funcanim": [12, 14], "function": [2, 10], "g": [2, 10], "gauss": 11, "gauss_legendre_step": 9, "gaussian": 2, "gener": 5, "geomspac": 6, "get": [6, 12], "get_cmap": 8, "get_real_method": 12, "ghost": 5, "give": 11, "go": 11, "grid": [4, 5, 6, 8], "guess": 9, "h": [6, 9, 10], "h2": 10, "h264": 12, "h2o": 10, "h_2": 10, "h_2o": 10, "ha": [2, 5, 12], "han": [13, 14], "hand": 8, "here": 9, "hold": 12, "horizont": 2, "how": 11, "howev": 5, "html": [12, 14], "html5": 12, "i": [2, 5, 6, 7, 8, 10, 11, 12, 13], "iT": 13, "idex": 12, "ij": 13, "ikh": 6, "illustr": 15, "imag": 6, "imax": 6, "implicit": [10, 11], "import": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "includ": 5, "increas": 12, "index": [6, 13], "ineg": 12, "inf": 10, "infer": 12, "inform": 11, "initi": [4, 5, 7, 8, 9, 10], "inlin": [4, 5, 12], "instationnair": [4, 5], "int": 9, "integr": [7, 8, 9, 10], "intens": 5, "intermedi": 9, "intern": 5, "interv": [12, 14], "intial": [4, 5], "introduct": 11, "invers": 10, "involv": 10, "ipykernel_2153": 12, "ipython": 12, "is_avail": 12, "iter": [5, 9], "ivp": 8, "j": [5, 10, 12], "jacobian": 10, "jshtml": [12, 14], "k": [5, 6, 8, 12, 14], "k1": [9, 10], "k2": [9, 10], "k3": 10, "k4": 10, "k_1": 10, "k_2": 10, "k_3": 10, "k_4": 10, "kei": [6, 8], "keyerror": 6, "kh": 6, "kmax": 14, "known": 2, "kutta": 11, "l": [4, 5, 12], "la": [4, 5], "label": [2, 6, 9, 10], "labels": 2, "lambda": [2, 6, 14], "largest": 10, "last": [5, 10, 12], "le": [4, 5], "lead": 10, "left": 5, "legend": [2, 6, 9, 10], "legendr": 11, "len": 2, "length": 12, "let": 10, "level": 8, "lib": 12, "librari": 10, "like": 10, "linalg": [10, 12], "line": [2, 4, 10, 12, 14], "line2d": [2, 4], "linear": [2, 8], "linestyl": [9, 10], "linewidth": [5, 8], "linspac": [2, 4, 5, 10, 12, 13, 14], "lin\u00e9air": [4, 5], "log": 10, "loglog": 6, "logspac": 10, "look": 10, "loop": 14, "m": [4, 5, 12], "m4v": 12, "made": 8, "mai": [5, 7, 10], "major": 11, "make": 5, "manual": 12, "map": 2, "mapf": [13, 14], "markers": 12, "matder": 2, "math": 8, "matmul": 12, "matplotlib": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "matric": 12, "matrix": [2, 10], "max": [10, 14], "max_eigenvalu": 10, "max_fram": 12, "maximum": 2, "maxmod": 12, "mechan": 10, "mesh": 2, "meshgrid": [8, 13, 14], "meth": 6, "method": [2, 6, 10, 11, 12], "might": 10, "mode": 12, "more": [5, 11], "most": [10, 12], "moviewrit": 12, "moviewriterregistri": 12, "mpl": 12, "much": 10, "must": [5, 12], "n": [2, 5, 8, 12], "n_": 5, "n_i": 5, "n_step": 9, "name": [8, 10, 12], "nameerror": 10, "ndof": [4, 5], "necessari": 10, "necessarili": 5, "neg": 12, "neumann": 11, "newaxi": 14, "newf": 2, "nit": 5, "non": 2, "none": 12, "nonlinear": 9, "nonlinear_system": 9, "np": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "nt": [13, 14], "number": [5, 12], "numer": [8, 12], "numpi": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "numtoolkit": 2, "nwave": 14, "nx": 14, "o": [2, 10, 12], "o2": 10, "o_2": 10, "obj": 12, "obtain": 5, "obviou": 5, "od": [8, 9, 10], "off": 11, "offer": 2, "often": 2, "oh": 10, "onc": 14, "onli": [2, 5, 10], "order": [2, 5, 8, 12], "ordinari": 10, "origin": 5, "oscil": 5, "other": 10, "out": 11, "over": 10, "own": 2, "p": [2, 8], "packag": 12, "page": 11, "param": 2, "partial": [4, 5, 10], "particular": 11, "pass": 12, "path": 12, "pcolormesh": 14, "perf_count": 12, "perform": 9, "period": 5, "physic": 5, "pi": [4, 5, 6, 12, 13, 14], "plai": 5, "plot": [2, 4, 5, 9, 10, 12, 14], "plotn": 2, "plt": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "pltanim": 14, "pod": 11, "pod_cov": 12, "pod_svd": 12, "point": [2, 5, 6, 9, 10, 12], "possibl": 5, "possibli": 12, "post": 15, "pour": [4, 5], "precis": 5, "prefer": 2, "prepar": 9, "previou": 8, "print": [6, 12], "print_method": 12, "problem": [7, 8, 10, 12], "probl\u00e8m": [4, 5], "process": [12, 15], "product": 2, "properti": 7, "proport": 2, "propos": 5, "py": 12, "pyplot": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "python": 5, "python3": 12, "p\u00e9riodiqu": [4, 5], "q": [4, 5, 6, 8], "q0": [4, 5], "q_0": [4, 5, 8], "q_i": 5, "qinit": [4, 5], "qnew": 5, "qquad": 5, "quantit\u00e9": [4, 5], "quick": 6, "r": [8, 12], "r_": 8, "radic": 10, "rais": 12, "rand": 14, "random": 14, "rang": [2, 5, 6, 8, 9, 12], "rate": 10, "rc": 14, "rcparam": [2, 12], "re": 10, "reaction": 10, "real": [6, 9], "reason": 5, "recent": [10, 12], "recommend": 2, "redefin": 10, "refin": 9, "reflect": 14, "remain": 5, "repeat": 14, "represent": 2, "request": 12, "requir": 10, "reset": 10, "resolv": 10, "respect": 10, "result": [9, 10], "return": [2, 4, 5, 6, 8, 9, 10, 12, 13, 14], "revers": 12, "rh": 6, "right": [2, 5, 8], "rightarrow": 10, "rk": 8, "rk45": 10, "rm": [4, 5], "rmode": 12, "robertson": 10, "robertson_jacobian": 10, "robertson_kinet": 10, "robetson": 10, "robust": 10, "roll": 6, "root": 12, "rotat": 2, "row": 11, "row2": 9, "row2_step": 9, "rung": 11, "runtimeerror": 12, "r\u00e9soudr": [4, 5], "s2": 12, "sampl": [2, 9, 11], "save": 12, "save_count": 12, "scale": 10, "scheme": 11, "scipi": [10, 12], "second": [2, 12], "section": 15, "see": 11, "self": 12, "semilogi": [2, 12], "set": [8, 9, 10], "set_titl": 2, "set_xlabel": [2, 6], "set_xlim": 14, "set_ydata": [12, 14], "set_ylabel": [2, 6], "set_ylim": [6, 13, 14], "sever": [5, 8, 15], "shade": 14, "shape": [12, 14], "shift": 6, "should": 8, "show": [5, 8, 9, 10, 11, 12], "side": 8, "sig": 14, "sigma": 5, "signal": 15, "significantli": 10, "simplifi": 10, "simul": 5, "sin": [4, 5, 12, 13, 14], "sinc": 2, "singl": 9, "singular": 12, "site": 12, "size": [2, 5, 12, 13], "skip": 14, "slower": 10, "small": 11, "snapshot": 12, "so": 12, "sol_explicit": 10, "sol_implicit": 10, "sol_implicit_robertson": 10, "sol_robertson": 10, "solut": [2, 5, 8, 9, 10], "solv": [8, 10], "solve_ivp": 10, "solver": 9, "some": [5, 10, 11, 12], "sort": 12, "sp": 2, "space": [5, 11], "speci": 10, "spectral": 11, "speed": 12, "splin": 12, "spop": 2, "sqrt": [2, 9, 12], "squar": 12, "squeez": 12, "sqval": 12, "stabil": [5, 10], "stage": [5, 8, 9], "state": 10, "stencil": 6, "step": [5, 8, 9, 10], "stiff": [5, 11], "storag": 9, "str": [5, 12], "structur": 11, "subplot": [2, 6, 8, 10, 12, 13, 14], "suivant": [4, 5], "sum": [2, 8], "superimpos": 5, "suppress": 12, "suptitl": 12, "sur": [4, 5], "sval": 12, "svd": [12, 15], "symmetr": 12, "syntax": 5, "system": [7, 9, 10], "t": [4, 5, 8, 9, 10, 12, 13, 14], "t0": 9, "t_eval": 10, "t_valu": 9, "tab10": 8, "tag": 12, "taylor": 8, "tcoef": 12, "techniqu": 15, "temp": 12, "tempor": 5, "temporari": 5, "term": 5, "test": 9, "text": [5, 10], "tf": [4, 9, 12], "th": 8, "than": 10, "thank": 2, "theoret": [2, 8], "therefor": 5, "thi": [5, 8, 10, 11, 12, 15], "thispod": 12, "through": 12, "tic": 12, "tight_layout": 10, "time": [5, 7, 9, 10, 11], "time_steps_explicit": 10, "time_steps_implicit": 10, "time_steps_implicit_robertson": 10, "titl": [5, 9, 10], "titles": 2, "tmp": 12, "tmpdir": 12, "to_html5_video": 12, "to_jshtml": 12, "toc": 12, "topic": 11, "traceback": [10, 12], "transform": 15, "transport": 5, "transport\u00e9": [4, 5], "transpos": 12, "true": [2, 12, 14], "truncat": 11, "tt": [12, 14], "ttot": 14, "two": 9, "type": 11, "u": [12, 14], "uconv": 14, "unbound": 12, "unit": 10, "unknown": 6, "unlik": 2, "up": 8, "updat": [2, 9, 12], "upwind": 11, "upwind1": 6, "upwind3": 6, "us": [2, 8, 9, 10], "userwarn": 12, "v": 12, "v1": 2, "valu": [2, 7, 8, 9, 12], "vari": 10, "variabl": 2, "vastli": 10, "vector": 2, "veloc": [4, 5], "venv": 12, "visibli": 5, "von": 11, "vt": 12, "warn": 12, "we": [5, 12], "weight": 8, "well": 11, "what": 5, "where": 10, "which": [2, 12], "whose": [2, 5], "work": 12, "would": 9, "writer": 12, "written": 5, "x": [2, 4, 5, 8, 12, 14], "x0": [12, 14], "x_uni": 2, "xi": [4, 5], "xlabel": [9, 10], "xscale": 10, "xx": [12, 14], "y": [8, 9, 10], "y0": [9, 10], "y1": [9, 10], "y2": [9, 10], "y3": 10, "y4": 10, "y5": 10, "y6": 10, "y_1": 10, "y_2": 10, "y_3": 10, "y_4": 10, "y_5": 10, "y_6": 10, "y_gauss": 9, "y_next": 9, "y_row2": 9, "y_values_gauss": 9, "y_values_row2": 9, "yet": 2, "ylab": 6, "ylabel": [9, 10], "you": [5, 11], "yscale": 10, "zero": [2, 5, 10, 12, 13], "zip": [6, 14], "\u00e0": [4, 5], "\u00e9volut": [4, 5]}, "titles": ["Finite Difference Method", "Spectral Collocation Methods", "Spectral discretization", "Introduction", "Introduction", "Upwind and Central scheme", "von Neumann analysis", "Introduction", "Runge-Kutta explicit methods", "ROW and Gauss-Legendre implicit method", "Stiffness", "Welcome to your Jupyter Book", "POD of time signal", "FFT of truncated signals", "2D FFT of space/time signal", "Introduction"], "titleterms": {"2d": 14, "analysi": 6, "basic": 11, "book": 11, "central": 5, "chebyshev": 2, "colloc": 1, "combust": 10, "condit": 4, "convect": 11, "curv": 2, "deriv": 2, "differ": 0, "discret": [2, 11], "dynam": 11, "equat": 11, "et": 4, "euler": 5, "explicit": [5, 8], "fft": [13, 14], "finit": 0, "fit": 2, "gauss": 9, "hydrogen": 10, "implicit": 9, "initial": 4, "integr": [5, 11], "introduct": [3, 4, 7, 15], "jupyt": 11, "kinet": 10, "kutta": 8, "lax": 5, "legendr": 9, "method": [0, 1, 5, 8, 9], "neumann": 6, "oper": 2, "oxygen": 10, "pod": 12, "process": 11, "row": 9, "rung": 8, "scheme": 5, "signal": [11, 12, 13, 14], "solut": 4, "space": 14, "spectral": [1, 2], "stiff": 10, "system": 11, "th\u00e9oriqu": 4, "time": [12, 14], "truncat": 13, "upwind": 5, "valid": 2, "von": 6, "welcom": 11, "wendroff": 5, "your": 11}})