package com.example.pf.my_tfandroid;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Vector;

/**
 * Created by pf on 3/25/18.
 */

public class TF_Caculation_Thread extends Thread {

    static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";
    static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    static final String INPUT_NODE = "input";
    static final String OUTPUT_NODE = "output";
    static final int inputSize = 224;
    static final int imageMean = 117;
    static final int imageStd = 1;
    static final int numClasses = 2000;

    private Bitmap bitmap;
    private Bitmap resized_bitmap;              //resized_bitmap which has a shape of inputSize X inputSize
    private String[] OUTPUT_NODES = new String[] {OUTPUT_NODE};
    private TensorFlowInferenceInterface inferenceInterface=null;
    private Handler threadHandler;
    private Vector<String> labels = new Vector<>();

    /**Constructure Function
     * @param btm           a bitmap from camera
     * @param assetManager  ok
     * @param mHandler      a Handler to send message to the UI thread to finish task which update the UI
     * */
    TF_Caculation_Thread(Bitmap btm, AssetManager assetManager, Handler mHandler){
        super();
        bitmap = btm;
        threadHandler = mHandler;
        BufferedReader br =null;
        String actualFilename = LABEL_FILE.split("file:///android_asset/")[1];
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while((line = br.readLine()) != null){
                labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    /**This function is to fill a bitmap into a square
     *@param srcBmp       the source of the bitmap
     *@param cfg          the Config of the original bitmap, from the Class Bitmap.Config
     * */
    private static Bitmap createSquaredBitmap(Bitmap srcBmp, Bitmap.Config cfg) {
        int dim = Math.max(srcBmp.getWidth(), srcBmp.getHeight());
        Bitmap dstBmp = Bitmap.createBitmap(dim, dim, cfg);

        Canvas canvas = new Canvas(dstBmp);
        canvas.drawColor(Color.WHITE);
        canvas.drawBitmap(srcBmp, (dim - srcBmp.getWidth()) / 2, (dim - srcBmp.getHeight()) / 2, null);

        return dstBmp;
    }

    /**This function classify a bitmap and return an array of result of caculation
     * @param btm a bitmap need to be classied
     * @return      an array of result
     * */
    public float[] classify(Bitmap btm){
        int[] intValues = new int[inputSize * inputSize] ;
        float[] floatValues = new float[inputSize*inputSize*3];
        float[] output_array = new float[numClasses];
        Bitmap square_bitmap = createSquaredBitmap(btm, btm.getConfig());

        resized_bitmap = Bitmap.createBitmap(square_bitmap,0,0,inputSize,inputSize);
        resized_bitmap.getPixels(intValues, 0, resized_bitmap.getWidth(), 0, 0, resized_bitmap.getWidth(), resized_bitmap.getHeight());

        //preprocess the bitmap, normalize the data and make it to the suitable shape to suit the input tensor
        for (int i = 0; i < inputSize*inputSize; ++i) {
            final int val;
            val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        inferenceInterface.feed(INPUT_NODE, floatValues, 1, inputSize, inputSize, 3);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE,output_array);
        return  output_array;
    }

    @Override
    public void run(){
        float[] result_array ;
        float max_value=0;
        int max_index=0;

        try{
            result_array = classify(bitmap);
            //get the index of most probability of classify
            for (int i=0; i< result_array.length; i++) {
                if(result_array[i] > max_value){
                    max_value= result_array[i];
                    max_index = i;
                }
            }

            Message msg_text = new Message();
            msg_text.what = 0;
            msg_text.obj = labels.get(max_index);       //translate into human readable string
            threadHandler.sendMessage(msg_text) ;       //send the text message, it will show the result of classify

            Message msg_image = new Message();
            msg_image.what = 1;
            msg_image.obj = resized_bitmap;
            threadHandler.sendMessage(msg_image);       //sned the bitmap message, it will show the original image

        }catch(Exception e){
            e.printStackTrace();
            Log.e("test","err");
        }
    }
}
