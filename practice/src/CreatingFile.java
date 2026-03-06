import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
public class CreatingFile {
    public static void main(String[] args){
        try{
            File file = new File("test.txt");
            if(file.createNewFile()){
                System.out.println("File created");
            }else{
                System.out.println("File already exists");
            }

            //2 write to the file
            FileWriter writer = new FileWriter("test.txt");
            writer.write("Hello Bimal");
            writer.close();
            System.out.println("Write successfully");

            //3
            File file1 = new File("test.txt");
            Scanner sc = new Scanner(file1);

            while(sc.hasNextLine()) {
                System.out.println(sc.nextLine());
            }
            sc.close();
        }catch (IOException e){
            System.out.println("Error occurred");
        }
    }

}
