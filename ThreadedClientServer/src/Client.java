import java.io.*;
import java.net.*;

public class Client {
    public static void main(String[] args){
        try{
            // Connect to server , localhost = own computer , port = 5000
            Socket socket = new Socket("localhost",5000);
            //Client  ---> pipe ---> Server , a pipe to recieve data
            //Server  ---> pipe ---> Client , pipe to send data
            BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
            // BufferedReader is used for read text efficiently
            // PrintWriter out = new PrintWriter(socket.getOutputStream(),true);
            PrintWriter out = new PrintWriter(socket.getOutputStream(),true);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(socket.getInputStream())  // bytes → characters , 01001000 → H
            );
            System.out.print("Enter message...");
            String msg = input.readLine();
            out.println(msg);
            String response = in.readLine();
            System.out.println("Server: "+ response);
            socket.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
