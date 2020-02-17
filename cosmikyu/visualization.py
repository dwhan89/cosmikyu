from visdom import Visdom
import numpy as np

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"
DEFAULT_BASEURL = '/'


class VisdomPlotter(object):
    def __init__(self, env_name='main', port=DEFAULT_PORT, server=DEFAULT_HOSTNAME, base_url=DEFAULT_BASEURL):
        self.viz = Visdom(port=port, server=server, base_url=base_url)
        assert(self.viz.check_connection(timeout_seconds=3))

        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, title='', xlabel='epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=x, Y=y, env=self.env, opts=dict(
                legend=[split_name],
                title=title,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=x, Y=y, env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')
