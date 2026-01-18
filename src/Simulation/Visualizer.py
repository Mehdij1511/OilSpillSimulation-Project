import matplotlib.pyplot as plt
import cv2

class Visualizer:
    def __init__(self, triangles, config, outputdir):
        """
        Initializes the Visualizer with the given triangles, configuration, and output directory.
        Args:
            triangles (list): A list of all traingles in the mesh.
            config (dict): Configuration settings for the visualization.
            outputdir (str): The directory where the output files will be saved.
        """
        self.triangles = triangles
        self.config = config
        self.outputfolder = outputdir


    def create_plot(self, oil_distribution, time, output_path):
        """
        Creates a plot of the oil distribution over the triangles and saves it to a file.
        Args:
            oil_distribution (list): The oil distribution values for each triangle.
            time (float): The current time of the simulation, used for the plot title.
            output_path (str): The file path where the plot image will be saved.
        """
        plt.figsize = (10, 8)

        for triangle in self.triangles:
            x = [p[0] for p in triangle.points]  # Vector 1
            y = [p[1] for p in triangle.points]  # Vector 2
            x.append(x[0])  # Go back to start to close triangle
            y.append(y[0])

            plt.fill(x, y, color=plt.cm.viridis(oil_distribution[triangle.idx]))


        plt.plot(0.35, 0.45, 'r+', label='Oil Source')
        plt.gca().add_patch(plt.Rectangle(
            (self.config.borders[0][0], self.config.borders[1][0]), self.config.borders[0][1], self.config.borders[1][1],
            fill=False, color='red',
            linestyle='--', label='Fishing Grounds'
        ))


        plt.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(0, 1),
            cmap=plt.cm.viridis
        ), ax=plt.gca(), label='Oil Amount')


        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.title(f'Oil Distribution at t = {time:.3f}')
        plt.legend()
        
        plt.savefig(output_path)
        plt.close()


    def create_animation(self, images, freq):
        """
        Creates an animation from a list of images and saves it as a video file.
        Args:
            images (list): List of file paths to the images to be included in the animation.
            freq (int): Frame rate / number of frames set as the writeFrequency in config.
        """
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(self.outputfolder / 'output.avi', fourcc, freq, (width, height)) # ADD TO CORRECT OUTPUT PATH AND NAME

        for image in images:
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()
