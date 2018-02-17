package javacnn.mnist;

import java.io.File;
import java.io.IOException;

public class ReadMnistTest
{
	public static void main( final String[] args ) throws IOException
	{
		final File imagesFile = new File( "data/train-images-idx3-ubyte.gz" );
		final File labelsFile = new File( "data/train-labels-idx1-ubyte.gz" );

		final ReadMnist readMnist = new ReadMnist( labelsFile, imagesFile );

		readMnist.createDataset();
	}
}
