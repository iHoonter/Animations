import numpy as np
import matplotlib
import matplotlib.figure
import matplotlib.artist
import matplotlib.text as txt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable, Tuple, Iterable, List

class _xy_config:
    '''
    Dynamic data class for more readible settings that can be tailored specifically to the x or y axes
    '''
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    def set(self, x=None, y=None):
        if x != None:
            self.x = x
        if y != None:
            self.y = y

class artist_updater:
    def __init__(self,
                 init_func: Callable[['plotter', dict], matplotlib.artist.Artist], init_func_args: dict = {},
                 update_func: Callable[['plotter', int, dict], matplotlib.artist.Artist] = None, update_func_args: dict = {},
                 plotter_instance = None) -> None:
        
        # Save functions and args
        self._init_function = init_func
        self._init_args = init_func_args
        self._update_function = update_func
        self._update_args = update_func_args

        # A plotter instance must be assigned, but this will be done by the plotter object, so don't worry about it
        self._plotter_instance = plotter_instance
        # An artist will be created upon callling the init_artist function
        self.artist = None

    def set_plotter(self, instance: 'plotter'):
        self._plotter_instance = instance

    def init_artist(self) -> matplotlib.artist.Artist:
        # Check to make sure the plotter instance was set
        if self._plotter_instance == None:
            raise RuntimeError(f"artist updater {self.__name__} did not get its plotter instance set")
        
        # Run the init function and store the artist
        self.artist = self._init_function(self._plotter_instance, **self._init_args)

        return self.artist

    def update_artist(self, frame: int) -> matplotlib.artist.Artist:
        # Check to make sure the plotter instance was set
        if self._plotter_instance == None:
            raise RuntimeError(f"artist updater {self.__name__} did not get its plotter instance set")
        
        # Run the update function and return artist
        return self._update_function(self._plotter_instance, frame, **self._update_args)


#-------------------#
# DEFAULT FUNCTIONS #
#-------------------#

#__Default data artist functions__#

def default_data_artist_init(instance: 'plotter', xy_data: Tuple[np.ndarray, np.ndarray]) -> matplotlib.artist.Artist:
    '''
    Default data artist initializer, only used if provided xy data instead of an artist updater
    '''
    instance.x_data, instance.y_data = xy_data

    # Scale the axes to fit data if the autofit setting is selected
    if instance.autofit:
        # Scale axis to limits of data
        instance.scale_axes( (np.min(instance.x_data), np.max(instance.x_data)), (np.min(instance.y_data), np.max(instance.y_data)) )

    # Set the number of frames based on data length
    instance.frames = len(instance.x_data)

    # Create and initialize artist on axes. Initial artist is empty.
    # note: kwargs not handled by plotter are passed onto the pyplot plot method
    data_artist = instance.ax.plot([], [], **instance.kwargs)[0]

    return data_artist

def default_data_artist_update(instance: 'plotter', frame: int) -> matplotlib.artist.Artist:
    '''
    Default data artist updater. This simply sets new data on the data artist by slicing the data range.
    If this simple approach doesn't work for the animation wanted then you must provide an update function in an
    artist_updater object at plot initialization.
    '''
    instance.data.artist.set_data(instance.x_data[:frame], instance.y_data[:frame])
    return instance.data.artist


#__Default annotation artist functions__#

def default_annotation_artist_init(instance: 'plotter', annotation: txt.Annotation | str | bool) -> txt.Annotation:
    '''
    Default annotation initializer.
    
    - annotation: If a string, then that string is used as the text parameter in an auto-generated annotation.
                    If an annotation object, then no default annotation is created, and instead that object is used.
    '''
    def _create_default_annotation(text: str):
        return instance.ax.annotate(text=text,
                                    xy=(0, 0),                                          # Initial position
                                    xytext=(10,-10),                                     # Offset annotation (10,10) pts from last point plotted, left+bottom aligned
                                    textcoords='offset points',
                                    ha='left',
                                    va='bottom',
                                    bbox=dict(boxstyle='round', fc='w', ec='black'),    # Bounding box, white background color, black border
                                    animated=True,                                      # For blitting
                                    visible=False
                                    )
    # Match the annotation type
    match annotation:
        case str():
            annotation_artist = _create_default_annotation(annotation)
        case txt.Annotation():
            annotation_artist = annotation
        case True:
            # No text or annotation object provided, default name is "plot"
            annotation_artist = _create_default_annotation("plot")
        case _:
            raise RuntimeError("in plotter, 'annotation' must one of: artist_updater, matplotlib.text.Annotation, string, bool")
        
    # Hide annotation initially
    annotation_artist.set_visible(False)

    # Return artist
    return annotation_artist

def default_annotation_artist_update(instance: 'plotter', frame) -> txt.Annotation:
    '''
    Default annotation updater. Moves the annotation to the last point plotted for each frame.
    '''
    if frame > len(instance.x_data) - 1:       # Check to make sure within data limits (For plot groups)
        frame = len(instance.x_data) - 1
    
    # Move annotation
    instance.annotation.artist.xy = (instance.x_data[frame], instance.y_data[frame])

    return instance.annotation.artist             # Return artist for blitting


class plotter:
    #-------------------#
    # UTILITY FUNCTIONS #
    #-------------------#

    def scale_axes(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        '''
        Set axis xy limits to given range, if the current axis is larger than data size then don't change axis limits
        '''
        # Get current axes limits
        cur_xlims = self.ax.get_xlim()
        cur_ylims = self.ax.get_ylim()

        # Add padding to new limits (Based off of padding setting in kwargs)
        tot_x = xlim[1] - xlim[0]     # Get the total range for x and y values
        tot_y = ylim[1] - ylim[0]

        # Padding
        if self.padding.x is False:
            x_pad = 0
        else:
            x_pad = (tot_x*self.padding.x - tot_x)/2  # Padding ammount to add is the scaled size of the x range subtract the original size. Halved because padding is added and subtracted.
        
        if self.padding.y is False:
            y_pad = 0
        else:
            y_pad = (tot_y*self.padding.y - tot_y)/2  

        # Set new axis limits, based on whatever is larger
        self.ax.set(xlim = [min(xlim[0] - x_pad, cur_xlims[0]), 
                            max(xlim[1] + x_pad, cur_xlims[1])],
                    ylim = [min(ylim[0] - y_pad, cur_ylims[0]),
                            max(ylim[1] + y_pad, cur_ylims[1])])
        

    #------------------------------#
    # CONSTRUCTOR AND DATA SETTING #
    #------------------------------#

    def __init__(self, 
                 # Provide data
                 # Data can come in two forms: x & y data or an artist updater object. 
                 # xy data will be used in constructing a default artist updater. 
                 data: Tuple[np.ndarray, np.ndarray] | artist_updater | dict, 
                 
                 # Annotation options (optional)
                 # Artist updater: (provided if custom annotations are wanted)
                 # matplotlib.text.Annotation object: Default annotation updater created for this object to use in annotations
                 # String: Default annotation created, string is text displayed
                 # Bool: True-Create a fully default annotation, False-No annotation made
                 annotation: artist_updater | txt.Annotation | str | bool = False,

                 # Manually set the number of frames to be played in animation (Don't need to set if not using custom update functions for the data artist)
                 # NOTE: It is recommended you set this inside of your data artist_update.init function
                 frames = 0,

                 # Manually set figure and axes (optional)
                 figure: matplotlib.figure.Figure = None,
                 axes: plt.Axes = None,

                 **kwargs) -> None:

        # Set data artist updater
        self._get_data_updater(data)

        # Set annotation artist updater
        self._get_annotation_updater(annotation)

        # Set frames
        self.frames = frames
        
        # Get figure and axes
        if isinstance(figure, matplotlib.figure.Figure) and isinstance(axes, plt.Axes):
            self.ax = axes
            self.fig = figure
        else:
            self.fig, self.ax = plt.subplots()

        # Retain kwargs
        self.kwargs = kwargs

        # Handle key-word arguments
        self._handle_kwargs()

    def _get_annotation_updater(self, anno):
        '''
        Artist updater: (provided if custom annotations are wanted)
        matplotlib.text.Annotation object: Default annotation updater created for this object to use in annotations
        String: Default annotation created, string is text displayed
        Bool: True-Create a fully default annotation, False-No annotation made
        '''
        if anno is False:
            self._annotated = False
        elif isinstance(anno, artist_updater):
            self._annotated = True
            self.annotation = anno
            # Set plotter instance
            self.annotation.set_plotter(self)
            # Check to see if a default updater needs to be provided (idk why you would make an updater object for an annotation without an updater, but still)
            if self.annotation._update_function == None:
                self.annotation._update_function = default_annotation_artist_update
        else:
            self._annotated = True
            self.annotation = artist_updater(init_func=default_annotation_artist_init, init_func_args=dict(annotation=anno),
                                              update_func=default_annotation_artist_update,
                                              plotter_instance=self)            

    def _get_data_updater(self, data):
        '''
        Function to unpack all data into an artist_updater that can be used by the rest of the functions.
        Acceptable formats:

        - Tuple[np.ndarray, np.ndarray]
            A tuple of x and y data to use in plotting/animations. This will be used to create a default artist updater
        - artist_updater
            An already created artist_updater with all associated data. Used for more custom behavior.
        - dict (Not implemented right now)
            A dictionary of kwargs for an artist updater. The kwargs are passed to _unpack_artist_updater_dict(). If 
            non-viable kwargs are passed then a RuntimeError() is raised
        '''
        
        # Identify how data was passed
        match data:
            case tuple():
                # Case for passing xy data
                self.data = artist_updater(init_func=default_data_artist_init, init_func_args=dict(xy_data=data),
                                           update_func=default_data_artist_update,
                                           plotter_instance=self)
            case artist_updater():
                self.data = data
                # Set the plotter instance of the artist updater
                self.data.set_plotter(self)
                # Check to see if an update function was provided, if so then all information is present, if not provide a default. 
                if self.data._update_function == None:
                    self.data._update_function = default_data_artist_update

            case _:
                raise RuntimeError(f"data passed to plotter is not a permitted type.")

    def _handle_kwargs(self):
        # Method to process all key word arguments. Those not processed are handed to the matplotlib artist
        # Defaults
        self.persist = False
        self.annotation_persist = False
        self.delay = 0
        self.autofit = True

        # Padding stuff
        # Default
        self.padding = _xy_config(x=1.05, y=1.05)
        # Parser
        def parse_padding(padding):
            match padding:
                case False:
                    self.padding.set(x=False, y=False)

                case int() | float():
                    padding = 1 + padding*0.01
                    self.padding.set(x=padding, y=padding)

                case (int() | float() | bool(), int() | float() | bool()):
                    temp = (0, 0)
                    for i in range(2):
                        match padding[i]:
                            case True:
                                temp[i] = 1.05
                            case False:
                                temp[i] = False
                            case _:
                                temp[i] = 1 + padding[0]*0.01
                    
                    self.padding.set(x=temp[0], y=temp[1])
                case _:
                    raise RuntimeError("plotter 'padding' kwarg must be of type: int | float | tuple(2x(int | float))")
        # Match each kwarg and save/do stuff
        keys = list(self.kwargs.keys())
        for key in keys:
            match key:
                case 'persist':
                    self.persist = self.kwargs.pop(key)
                case 'annotation_persist':
                    self.annotation_persist = self.kwargs.pop(key)
                # Padding kwargs
                case 'padding' | 'pad':
                    # Padding to add around borders (in percent)
                    parse_padding(self.kwargs.pop(key))
                case 'xpadding' | 'xpad':
                    # Padding on the xaxis
                    self.padding.x = 1 + self.kwargs.pop(key)*0.01
                case 'ypadding' | 'ypad':
                    # Padding on yaxis
                    self.padding.y = 1 + self.kwargs.pop(key)*0.01
                case 'delay':
                    # Add a delay (in seconds) to an animation after this plot has been completed and before starting the next plot
                    self.delay = self.kwargs.pop(key)
                case 'xlim':
                    # Manually set the x limits
                    self.ax.set_xlim(self.kwargs.pop(key))
                case 'ylim':
                    # Manually set the y limits
                    self.ax.set_ylim(self.kwargs.pop(key))
                case 'autofit':
                    # Bool to set if the axes will automatically scale or not
                    if not isinstance(self.kwargs['autofit'], bool):
                        # If not a bool raise an error
                        raise RuntimeError("plotter kwarg 'autofit' must be a bool")
                    else:
                        # Store setting
                        self.autofit = self.kwargs.pop(key)

    #---------------#
    # OTHER METHODS #
    #---------------#

    def update(self, frame):
        # Call all update functions
        # Call data updater
        self.data.update_artist(frame)

        # Annotation update
        if self._annotated:
            self.annotation.update_artist(frame)

        if frame == 0:              # Show artists on first update
            self.visible(True)

        # Return modified artists for blitting
        return self.get_artists()

    def get_artists(self):
        artists = [self.data.artist]
        if self._annotated:
            artists.append(self.annotation.artist)
        
        return artists

    def animation_init(self):
        # Use other init methods to prepare to be animated
        self.data.init_artist()
        if self._annotated:
            self.annotation.init_artist()

    def _data_artist_visible(self, visible=True):
        # Method to hide data artist during animations
        if self.data.artist == None:
            self.data.init_artist()

        # self._data_artist.set_data([], [])
        self.data.artist.set_visible(visible)

    def _annotation_artist_visible(self, visible=True):
        # Hide the annotation artist
        if self.annotation.artist == None:
            self.annotation.init_artist()
        self.annotation.artist.set_visible(visible)

    def visible(self, visible = True):
        # Hide all plot artists
        self._data_artist_visible(visible)
        if self._annotated:
            self._annotation_artist_visible(visible)

    def _expire(self):
        '''
        Method to call when sequential animation is finished, if the object has persist=True then the artists remain visible.
        Same goes for annotation_persist, but that applies only to the annotation artist
        '''
        self.visible(self.persist)
        if self._annotated:
            self._annotation_artist_visible(self.annotation_persist)

    def plot(self):
        self.data.init_artist()
        self.data.update_artist(self.frames)

    def animate(self):
        self.animation_init()
        self._animation = animation.FuncAnimation(fig=self.fig, func=self.update, interval=10, frames=self.frames, blit=True)


class animator:
    def __init__(self, plots: Iterable[plotter | Tuple[plotter, ...]], 
                 figure: matplotlib.figure.Figure = None, 
                 axes: matplotlib.pyplot.Axes = None, 
                 **kwargs) -> None:
        # Retain key-word args
        self.kwargs = kwargs

        # Get plots. 
        # In order to handle simultaneous tuple groups, turn every individual group into a singular tuple
        # this turns each plot item into an interable of plots. Those in tuple groups will be plotted at the same time.
        self.plots : List[Tuple[plotter, ...]] = []
        for plot in plots:                  # loop over all the plots and plot groups provided
            if isinstance(plot, Tuple):     # If a plot group
                self.plots.append(plot)         # Append plot group
            else:                           # If an individual plot
                self.plots.append((plot, ))  # Convert to single Tuple and append

        # Get figure and axes
        if figure != None and axes != None:
            self.fig, self.ax = figure, axes
        else:
            self.fig, self.ax = plt.subplots()

        #________Retrieving plot data________#
        # Get all needed data from the plots, and set needed data on plots so everything meshes. 
        self.frames = []
        self.artists = []

        for plot in self.plots:
            max_frames = 0
            for p in plot:          # Iterate over each plot in plot Tuple
                # Set each plot's figure and axes to match
                if p.ax != self.ax:
                    p.ax.remove()
                    p.ax = self.ax
                if p.fig != self.fig:
                    plt.close(p.fig)
                    p.fig = self.fig
                
                # Initialize plot artists and add plot artists for blitting
                p.animation_init()
                for artist in p.get_artists():
                    self.artists.append(artist)

                # Find the frames for each plot, for plot groups, find the maximum frames from the group
                if p.frames > max_frames:
                    max_frames = p.frames
            
            # Append the max frames from frame group
            self.frames.append(max_frames)

        # Initialize plot index, which keeps track of which plot is being animated
        self.plot_index = 0

        # Handle key word arguments
        self._handle_kwargs()

    def _handle_kwargs(self):
        # Method to process all key word arguments. Those not processed are handed to the animation object
        # Defaults
        self.delay = 0
        self.blit = True
        self.interval = 10
        self._saved = False
        # Match each kwarg and save/do stuff
        keys = list(self.kwargs.keys())
        for key in keys:
            match key:
                case 'delay':
                    # Add a delay (in seconds) to an animation after this plot has been completed and before starting the next plot
                    self.delay = self.kwargs.pop(key)
                case 'blit':
                    # Control if blit is on or off (on by default)
                    self.blit = self.kwargs.pop(key)
                case 'interval':
                    # Set animation interval
                    self.interval = self.kwargs.pop(key)
                case 'xlim':
                    # Set the xlimits of the axes (overrides plotter axis scaling)
                    self.ax.set_xlim(self.kwargs.pop(key))
                case 'ylim':
                    # Set the ylimits of the axes (overrides plotter axis scaling)
                    self.ax.set_ylim(self.kwargs.pop(key))
                case 'save':
                    # Properties for saving animation using the matplotlib save method
                    self._saved = True
                    # Pack save args for use later (save requires a dictionary)
                    if isinstance(self.kwargs['save'], dict):
                        self._save_args = self.kwargs.pop(key)
                    else:
                        raise RuntimeError("animator kwarg 'save' must be a dictionary of arguments to use in the animator.save method")


    def _frame_generator(self, fps: bool | int = False) -> int:
        '''
        Generator to create the frame iterable used by the animation object. Automatically yields the correct number of frames for each frame group and increments
        the plot index. Also yields the start and delay frames, the delay frames are important for delays, obviously.
        '''
        
        def yield_delay(fps: bool | int, delay: float):
            '''
            Calculates and returns the number of delay frames to yield, which changes based whether using an interval or fps count
            '''

            if fps is False:                                    # Case for interval timing
                return int(delay/(self.interval * 0.001))       # The number of frame updates to delay for is decided by the frame interval, which is the time between frame updates (in ms), and the delay time wanted (in s). 
            
            else:                                               # Case for fps timing
                return int(delay * fps)                         # delay frames [frames] = [frames/second] * [seconds to delay]


        # _Beginning of animation_
        yield 'start'

        # _Repeat loop for each plot group_
        for i, plot_group in enumerate(self.plots):
            # Get the max frames and max delay in the plot group
            max_frames = 0
            max_delay = 0
            for plot in plot_group:
                if plot.frames > max_frames:
                    max_frames = plot.frames
                if plot.delay > max_delay:
                    max_delay = plot.delay

            # yield incrementing frames, the number of frames is specified by the max frames in the plot group
            for frame in range(max_frames):
                yield frame

            # Delay for the max delay time specified in the plot group
            for _ in range(yield_delay(fps, max_delay)):
                yield 'delay'

            # __Frames for plot group completed__
            # You don't want the plots to expire at the end of the animation, so that the animation delay still shows useful plots
            # so this check if this is the last plot group
            if i != len(self.plots)-1: 
                # Update the plot index 
                self.plot_index += 1
                # expire current plot group
                for plot in plot_group:
                    plot._expire()

        # Animation finished, do animation delay (Delay between end of animation and start of next one)
        for _ in range(yield_delay(fps, self.delay)):
            yield 'delay'

    def _get_save_frames(self, fps: bool | int = False):
        '''
        Calculates the number of frames to save using the 'save' method
        '''
        tot_frames = 1

        def calc_delay_frames(fps: bool | int, delay: float):
            if fps is False:                                    # Case for interval timing
                return int(delay/(self.interval * 0.001))       # The number of frame updates to delay for is decided by the frame interval, which is the time between frame updates (in ms), and the delay time wanted (in s). 
            else:                                               # Case for fps timing
                return int(delay * fps)                         # delay frames [frames] = [frames/second] * [seconds to delay]


        # _Repeat loop for each plot group_
        for i, plot_group in enumerate(self.plots):
            # Get the max frames and max delay in the plot group
            max_frames = 0
            max_delay = 0
            for plot in plot_group:
                if plot.frames > max_frames:
                    max_frames = plot.frames
                if plot.delay > max_delay:
                    max_delay = plot.delay

            # Add max frames to total frames
            tot_frames += max_frames

            # Add delay frames to total
            tot_frames += calc_delay_frames(fps, max_delay)         

        # Add delay frames at end of animation
        tot_frames += calc_delay_frames(fps, self.delay)

        return tot_frames

    def sequential_update(self, frame):
        # Reset the animation index and animation at beginnning of animation
        if frame == 'start':
            # Reset index
            self.plot_index = 0
            # Reset artists
            for plot in self.plots:
                for p in plot:
                    p.visible(False)
            # Do nothing else
            return self.artists

        # If a delay frame, then do nothing
        if frame == 'delay':
            return self.artists
        
        # Save the current plot group
        cur_plots = self.plots[self.plot_index]

        # Update plot group artists
        for plot in cur_plots:
            plot.update(frame)

        return self.artists     # For blitting

    def animate(self):
        '''
        Starts the animation
        '''
        # Check if 'save' is true in kwargs, if so then create the self.save animation object instead
        if self._saved:
            self.save(**self._save_args)
        # Create the animation object
        self._anim = animation.FuncAnimation(fig=self.fig, func=self.sequential_update, frames=self._frame_generator, save_count=self._get_save_frames(), blit=self.blit, **self.kwargs, interval=self.interval)        

    def save(self, filename: str, writer: str = 'pillow', fps: int = None):
        # Default fps, calculated to provide the same speed as the interval speeed
        if fps is None:
            fps = int(1000/self.interval)   # fps is the inverse of interval/1000, since interval is in ms
        # Create animation object, but with frame generator in fps mode
        self._anim = animation.FuncAnimation(fig=self.fig, func=self.sequential_update, frames=self._frame_generator(fps), save_count=self._get_save_frames(fps), blit=self.blit, interval=self.interval, **self.kwargs)
        # Save the animation
        self._anim.save(filename=filename, writer=writer, fps=fps)





def main():         
    class spin_plot:
        def spin_plot_init(self, instance: plotter):
            self.center = (7.5, 7.5)
            self.radius = 7.5

            x1 = self.center[0]-self.radius
            x2 = self.center[0]+self.radius

            instance.scale_axes((x1, x2), (self.center[1]-self.radius, self.center[1]+self.radius))

            return instance.ax.plot((x1,x2), (0,0), '-r')[0]

        def spin_plot_update(self, instance: plotter, frame: int):
            x1 = self.radius*np.cos(np.radians(frame)) + self.center[0]
            x2 = self.radius*np.cos(np.radians(frame + 180)) + self.center[0]
            y1 = self.radius*np.sin(np.radians(frame)) + self.center[1]
            y2 = self.radius*np.sin(np.radians(frame + 180)) + self.center[0]

            instance.data.artist.set_data([x1, x2], [y1, y2])

            return instance.data.artist

    def anno_init(instance: plotter):
        return instance.ax.annotate("Point: (0, 0)",
                                    xy=(instance.x_data[0], instance.y_data[0]),
                                    xytext=(6, 12),
                                    textcoords='data',
                                    arrowprops=dict(arrowstyle='->')
                                    )

    def anno_update(instance: plotter, frame):
        x, y = (instance.x_data[frame], instance.y_data[frame])
        instance.annotation.artist.xy = (x,y)
        instance.annotation.artist.set(text = f"Point: ({x:.1f}, {y:.1f})")

    def eq1(scale: float):
        x = np.linspace(0, 15, 500)
        y = scale*np.exp(-(x/5))

        return (x, y)
    
    anno_artup = artist_updater(init_func=anno_init, update_func=anno_update)
    
    fig, ax = plt.subplots()

    circle_plotter = spin_plot()
    circle_plot_artup = artist_updater(init_func=circle_plotter.spin_plot_init, update_func=circle_plotter.spin_plot_update)
    circle_plot = plotter(circle_plot_artup, figure=fig, axes=ax, frames=360, delay=0.5, pad=30)
    
    plot1 = plotter(eq1(10), annotation="name", figure=fig, axes=ax, persist=True, annotation_persist=False, delay=1)
    plot2 = plotter(eq1(15), annotation=False, figure=fig, axes=ax)

    plot3 = plotter(eq1(12.5), annotation=anno_artup, figure=fig, axes=ax)

    ax.set_aspect('equal')


    anim = animator([circle_plot, (plot2, plot1), plot3], figure=fig, axes=ax, interval=5, delay=2)
    # anim.save("testing.mp4", writer="ffmpeg")
    anim.animate()
    plt.show()

if __name__ == "__main__":
    main()
