package com.example.pf.my_tfandroid;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Picture;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }
    static final int REQUEST_CODE_IMAGE_CAPTURE = 1;
    //static final String MODEL_FILE = "file:///android_asset/easy_pb.pb";
    //static final String INPUT_NODE = "input_node:0";
    //static final String OUTPUT_NODE = "output_node:0";
    static final String MODEL_FILE = "file:///android_asset/classify_image_graph_def.pb";
    static final String INPUT_NODE = "DecodeJpeg/contents:0";
    static final String OUTPUT_NODE = "softmax:0";
    static final int inputSize = 224;
    static final int imageMean = 117;
    static final int imageStd = 1;
    static final int numClasses = 2000;
    public ImageView imageView;
    public TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        if(requestCode == REQUEST_CODE_IMAGE_CAPTURE && resultCode == RESULT_OK){
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");

            TF_Caculation_Thread thread_tf = new TF_Caculation_Thread(imageBitmap, getAssets(), textView, imageView);
            thread_tf.start();
        }
    }

    public void takePhoto(View v){
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(takePictureIntent.resolveActivity(getPackageManager()) != null){
            startActivityForResult(takePictureIntent, REQUEST_CODE_IMAGE_CAPTURE);
        }
    }

}
